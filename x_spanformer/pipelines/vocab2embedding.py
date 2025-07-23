#!/usr/bin/env python3
"""
vocab2embedding.py

Transforms vocabulary files into seed embeddings and span candidates using the
approach described in Section 3.2 of the X-Spanformer paper. This pipeline
implements the unified algorithm for:

1. Soft probability computation via forward-backward algorithm
2. Seed embedding generation with vocabulary-aware initialization 
3. Multi-scale dilated convolutional contextualization
4. Vocabulary-informed span candidate enumeration

The implementation uses PyTorch with CUDA acceleration for efficient processing
of long sequences and large vocabularies.

Input Format:
- PretrainRecord format (dataset.jsonl): {"raw": "text", "type": "...", "meta": {...}}

This allows direct processing of dataset files from the Section 3.1 pipeline
without requiring separate sequence extraction steps.
"""
import argparse
import json
import math
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union

import torch
import torch.nn as nn
import yaml
import numpy as np

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.embedding.embedding_logging import setup_embedding_logging, get_embedding_logger
from x_spanformer.embedding.embedding_utils import (
    analyze_embedding_quality
)
from x_spanformer.pipelines.shared.jsonl_processor import load_pretrain_records
from x_spanformer.kernel import ConvEncoderKernel, validate_convolution_parameters

# Module-level logger that gets configured in main()
logger = None

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    if logger:
        logger.warning("SHUTDOWN SIGNAL RECEIVED - Finishing current sequence and exiting gracefully...")
        logger.warning(f"Signal: {signum}, Frame: {frame}")
    else:
        print("SHUTDOWN SIGNAL RECEIVED - Finishing current sequence and exiting gracefully...")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, signal_handler)


class UnigramLM(nn.Module):
    """
    Unigram Language Model implementing the forward-backward algorithm
    for soft piece probability computation as described in Section 3.2.1.
    """
    
    def __init__(self, vocab_dict: Dict[str, float], device: str = 'cuda'):
        """
        Initialize UnigramLM with vocabulary probabilities.
        
        Args:
            vocab_dict: Dictionary mapping pieces to their probabilities
            device: PyTorch device for computation
        """
        super().__init__()
        self.device = device
        
        # Create piece-to-index mapping
        self.piece_to_idx = {piece: i for i, piece in enumerate(vocab_dict.keys())}
        self.idx_to_piece = {i: piece for piece, i in self.piece_to_idx.items()}
        self.vocab_size = len(vocab_dict)
        
        # Store piece lengths for efficient matching
        self.piece_lengths = {piece: len(piece) for piece in vocab_dict.keys()}
        self.max_piece_length = max(self.piece_lengths.values()) if self.piece_lengths else 0
        
        # Convert probabilities to log space for numerical stability
        log_probs = [math.log(vocab_dict[piece]) for piece in self.piece_to_idx.keys()]
        self.log_piece_probs = torch.tensor(log_probs, device=device, dtype=torch.float32)
    
    def matches_at_position(self, sequence: str, pos: int, piece: str) -> bool:
        """Check if piece matches sequence starting at position pos."""
        if pos + len(piece) > len(sequence):
            return False
        return sequence[pos:pos + len(piece)] == piece
    
    def forward_backward(self, sequence: str) -> torch.Tensor:
        """
        Compute soft piece probabilities using forward-backward algorithm
        exactly as specified in Section 3.2.1, Equations (2), (3), and (4).
        
        Note: Paper uses 1-indexed positions, we use 0-indexed internally
        but follow the mathematical semantics precisely.
        
        Args:
            sequence: Input codepoint sequence as string
            
        Returns:
            Probability matrix P ∈ R^{T × |V|} where P[t,i] is the probability
            of piece u_i starting at position t (using paper's semantics)
        """
        T = len(sequence)
        V = self.vocab_size
        
        # Initialize forward probabilities α_t (Equation 2)
        # Paper: α_1 = 1, we use α_0 = 1 (0-indexed)
        alpha = torch.full((T + 1,), -float('inf'), device=self.device)
        alpha[0] = 0.0  # log(1) = 0
        
        # Forward pass: α_{t+1} = Σ_{u_i : match(x,t,u_i)} α_t * p(u_i) (Equation 2)
        for t in range(T):
            if alpha[t] == -float('inf'):
                continue
                
            # Check all vocabulary pieces that match at position t
            for piece_idx, piece in enumerate(self.idx_to_piece.values()):
                if self.matches_at_position(sequence, t, piece):
                    next_pos = t + len(piece)
                    if next_pos <= T:
                        # Accumulate probability: α_{t+|u_i|} += α_t * p(u_i) (log space)
                        alpha[next_pos] = torch.logsumexp(
                            torch.stack([alpha[next_pos], 
                                       alpha[t] + self.log_piece_probs[piece_idx]]),
                            dim=0
                        )
        
        # Initialize backward probabilities β_t (Equation 3)
        # Paper: β_{T+1} = 1, we use β_T = 1 (0-indexed)
        beta = torch.full((T + 1,), -float('inf'), device=self.device)
        beta[T] = 0.0  # log(1) = 0
        
        # Backward pass: β_t = Σ_{u_i : match(x,t,u_i)} p(u_i) * β_{t+|u_i|} (Equation 3)
        for t in range(T - 1, -1, -1):
            for piece_idx, piece in enumerate(self.idx_to_piece.values()):
                if self.matches_at_position(sequence, t, piece):
                    next_pos = t + len(piece)
                    if next_pos <= T and beta[next_pos] != -float('inf'):
                        # Accumulate probability: β_t += p(u_i) * β_{t+|u_i|} (log space)
                        beta[t] = torch.logsumexp(
                            torch.stack([beta[t],
                                       self.log_piece_probs[piece_idx] + beta[next_pos]]),
                            dim=0
                        )
        
        # Compute soft piece probabilities (Equation 4)
        # P_{t,i} = (α_t * p(u_i) * β_{t+|u_i|}) / α_{T+1} if match(x,t,u_i), else 0
        P = torch.zeros((T, V), device=self.device)
        
        normalization = alpha[T]  # α_{T+1} in paper notation
        if normalization == -float('inf'):
            get_embedding_logger('vocab2embedding').warning(f"Sequence cannot be segmented with given vocabulary")
            return P
        
        # Count non-zero probabilities for progress reporting
        prob_count = 0
        for t in range(T):
            for piece_idx, piece in enumerate(self.idx_to_piece.values()):
                if self.matches_at_position(sequence, t, piece):
                    next_pos = t + len(piece)
                    if next_pos <= T:
                        # Exact implementation of Equation (4)
                        log_prob = (alpha[t] + 
                                  self.log_piece_probs[piece_idx] + 
                                  beta[next_pos] - 
                                  normalization)
                        P[t, piece_idx] = torch.exp(log_prob)
                        if P[t, piece_idx] > 0:
                            prob_count += 1
        
        return P


class SeedEmbedder(nn.Module):
    """
    Seed embedding module implementing vocabulary-aware Xavier initialization
    as described in Section 3.2.2.
    """
    
    def __init__(self, vocab_dict: Dict[str, float], embed_dim: int, device: str = 'cuda'):
        """
        Initialize seed embedder with vocabulary-aware embeddings.
        
        Args:
            vocab_dict: Dictionary mapping pieces to their probabilities  
            embed_dim: Embedding dimension d
            device: PyTorch device for computation
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.vocab_size = len(vocab_dict)
        
        # Create embedding matrix with vocabulary-aware initialization
        self.embedding_matrix = nn.Parameter(
            torch.zeros(self.vocab_size, embed_dim, device=device)
        )
        
        # Initialize embeddings using Equations (5) and (6) from Section 3.2.2
        # W_emb[i,:] ~ N(0, σ²/√p(u_i))  where σ² = 2/(d + |V|) (Xavier-style scaling)
        pieces = list(vocab_dict.keys())
        probs = list(vocab_dict.values())
        
        # Calculate Xavier-style base variance (Equation 6)
        xavier_base_variance = 2.0 / (embed_dim + len(vocab_dict))
        
        for i, (piece, prob) in enumerate(zip(pieces, probs)):
            if len(piece) == 1:  
                # Single codepoint - standard Xavier initialization
                std = math.sqrt(xavier_base_variance)
            else:  
                # Multi-codepoint piece - frequency-scaled Gaussian (Equation 5)
                # σ²/√p(u_i) becomes std = √(σ²/√p(u_i)) = √(xavier_base_variance/√prob)
                std = math.sqrt(xavier_base_variance / math.sqrt(prob))
            
            with torch.no_grad():
                self.embedding_matrix[i].normal_(0, std)
    
    def forward(self, soft_probs: torch.Tensor) -> torch.Tensor:
        """
        Generate seed embeddings from soft probabilities.
        
        Args:
            soft_probs: Soft piece probabilities P ∈ R^{T × |V|}
            
        Returns:
            Seed embeddings H^0 = P * W_emb ∈ R^{T × d}
        """
        return torch.matmul(soft_probs, self.embedding_matrix)


class SpanCandidateGenerator:
    """
    Span candidate generator implementing vocabulary-informed filtering
    as described in Section 3.2.4.
    """
    
    def __init__(self, vocab_dict: Dict[str, float], tau_vocab: float = 1e-4, 
                 tau_comp: float = 1e-6, w_max: int = 64):
        """
        Initialize span candidate generator.
        
        Args:
            vocab_dict: Dictionary mapping pieces to their probabilities
            tau_vocab: Vocabulary alignment threshold
            tau_comp: Compositional potential threshold  
            w_max: Maximum span width
        """
        self.vocab_dict = vocab_dict
        self.tau_vocab = tau_vocab
        self.tau_comp = tau_comp
        self.w_max = w_max
        
        # Pre-compute sorted pieces for efficient matching
        self._sorted_pieces = sorted(vocab_dict.keys(), key=len, reverse=True)
        
        # Pre-compute vocabulary set for O(1) lookups
        self._vocab_set = set(vocab_dict.keys())
    
    def vocabulary_alignment(self, span_text: str) -> bool:
        """Check if span has high-probability vocabulary alignment."""
        if span_text in self._vocab_set:
            prob = self.vocab_dict[span_text]
            if not isinstance(prob, (int, float)):
                return False
                
            return prob >= self.tau_vocab
        return False
    
    def compositional_potential(self, span_text: str) -> bool:
        """Check if span has compositional segmentation potential using efficient greedy matching."""
        if not span_text or len(span_text) > 20:  # Skip very long spans
            return False
        
        pos = 0
        log_prob = 0.0
        
        while pos < len(span_text):
            # Find longest matching piece using sorted list (early termination)
            best_piece = None
            
            for piece in self._sorted_pieces:
                if len(piece) > len(span_text) - pos:
                    continue  # Piece too long
                if span_text[pos:pos + len(piece)] == piece:
                    best_piece = piece
                    break  # Found longest match, stop searching
            
            if best_piece is None:
                return False  # Cannot segment
            
            log_prob += math.log(self.vocab_dict[best_piece])
            pos += len(best_piece)
        
        return math.exp(log_prob) >= self.tau_comp
    
    def whitespace_coherent(self, span_text: str) -> bool:
        """
        Check if span represents coherent x-bar linguistic units.
        
        X-bar spans should represent complete linguistic units (full words, phrases)
        rather than arbitrary subword pieces. This ensures we move from subword 
        token-level representation to meaningful syntactic spans.
        """
        if not span_text:
            return False
        
        # Strip leading/trailing whitespace to get the core content
        stripped = span_text.strip()
        if not stripped:
            return False
        
        # X-bar spans should be complete words or phrases, not split words
        # Allow spans that:
        # 1. Are single complete words (no internal spaces)
        # 2. Are multi-word phrases (complete words separated by spaces)
        # 3. Don't start or end in the middle of alphanumeric sequences
        
        # Check if it's a single word
        if ' ' not in stripped:
            return True
        
        # For multi-word spans, ensure they don't split words
        # Original span should start and end at word boundaries in the context
        original_start_ok = span_text[0].isspace() or not span_text[0].isalnum()
        original_end_ok = span_text[-1].isspace() or not span_text[-1].isalnum()
        
        # Or the stripped version should be complete words
        words = stripped.split()
        all_complete_words = all(word.isalpha() or word.isalnum() for word in words if word)
        
        return (original_start_ok and original_end_ok) or all_complete_words
    
    def generate_candidates(self, sequence: str) -> List[Tuple[int, int]]:
        """
        Generate filtered span candidates following Algorithm 3 from Section 3.2.4.
        
        Optimized single-threaded implementation to avoid multiprocessing overhead
        and memory issues. Uses efficient early termination and batch processing.
        
        Args:
            sequence: Input codepoint sequence
            
        Returns:
            List of (start, end) candidate span positions (0-indexed, exclusive end)
        """
        T = len(sequence)
        candidates = []
        
        # Statistics for debugging
        vocab_aligned = 0
        comp_potential = 0
        whitespace_coherent = 0
        total_checked = 0
        
        # Single-threaded processing with early termination optimizations
        for i in range(T):
            # Check for shutdown signal
            if SHUTDOWN_REQUESTED:
                if logger:
                    logger.warning("Shutdown requested during candidate generation")
                break
                
            max_j = min(i + self.w_max, T)
            
            for j in range(i + 1, max_j + 1):
                total_checked += 1
                span_text = sequence[i:j]
                
                # Apply the three filtering criteria in order of computational cost
                # (fastest to slowest to maximize early termination)
                if self.vocabulary_alignment(span_text):
                    candidates.append((i, j))
                    vocab_aligned += 1
                elif self.whitespace_coherent(span_text):
                    candidates.append((i, j))
                    whitespace_coherent += 1
                elif self.compositional_potential(span_text):
                    candidates.append((i, j))
                    comp_potential += 1
        
        # Log statistics for debugging
        if logger:
            logger.debug(f"Candidate generation stats: {total_checked} spans checked, "
                        f"{len(candidates)} candidates ({vocab_aligned} vocab-aligned, "
                        f"{comp_potential} comp-potential, {whitespace_coherent} whitespace-coherent)")
        
        return candidates


class Vocab2EmbeddingPipeline:
    """
    Main pipeline class implementing the unified algorithm from Section 3.2.5.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the vocab2embedding pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Configure device from config with CUDA fallback to CPU
        processing_config = self.config.get('processing', {})
        base_device = processing_config.get('device', 'cuda')
        device_id = processing_config.get('device_id', 0)
        
        if base_device == 'cuda':
            if torch.cuda.is_available():
                self.device = f"cuda:{device_id}" if torch.cuda.device_count() > 1 else "cuda"
            else:
                get_embedding_logger('vocab2embedding').warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"
            
        get_embedding_logger('vocab2embedding').info(f"Initialized vocab2embedding pipeline on {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract all config parameters from nested structure
        architecture_config = config.get('architecture', {})
        span_config = config.get('span_generation', {})
        processing_config = config.get('processing', {})
        numerical_config = config.get('numerical', {})
        output_config = config.get('output', {})
        
        self.embed_dim = architecture_config.get('embed_dim', 512)
        self.dropout_rate = architecture_config.get('dropout_rate', 0.1)
        self.tau_vocab = span_config.get('tau_vocab', 1e-4)
        self.tau_comp = span_config.get('tau_comp', 1e-6)
        
        # Common parameter extraction
        self.max_sequence_length = processing_config.get('max_sequence_length', 512)
        self.batch_size = processing_config.get('batch_size', 64)
        self.epsilon = numerical_config.get('epsilon', 1e-12)
        self.max_piece_length = numerical_config.get('max_piece_length', 8)
        self.save_intermediate = output_config.get('save_intermediate', True)
        
        # Optional embedding controls (contextual embeddings H are ALWAYS saved as essential)
        self.save_seed_embeddings = output_config.get('save_seed_embeddings', False)  # Optional intermediate H⁰
        
        self.save_json_metadata = output_config.get('save_json_metadata', True)
        self.add_analysis = output_config.get('add_analysis', False)
        self.save_soft_probabilities = output_config.get('save_soft_probabilities', True)
        
        # Dynamic w_max calculation based on max_sequence_length and vocabulary
        # Set as half of max sequence length as computational bound
        self.w_max_bound = self.max_sequence_length // 2
        
        # Validate convolution parameters early
        conv_kernels = architecture_config.get('conv_kernels')
        conv_dilations = architecture_config.get('conv_dilations')
        
        if conv_kernels is None:
            raise ValueError("architecture.conv_kernels must be specified in configuration")
        if conv_dilations is None:
            raise ValueError("architecture.conv_dilations must be specified in configuration")
        
        # Use centralized validation from kernel package
        validate_convolution_parameters(conv_kernels, conv_dilations)
            
        get_embedding_logger('vocab2embedding').info(f"Validated convolution config: kernels={conv_kernels}, dilations={conv_dilations}")
        
        # Initialize components (will be set when vocabulary is loaded)
        self.unigram_lm: Optional[UnigramLM] = None
        self.seed_embedder: Optional[SeedEmbedder] = None  
        self.conv_encoder: Optional[ConvEncoderKernel] = None
        self.candidate_generator: Optional[SpanCandidateGenerator] = None
        
        return config
    
    def load_vocabulary(self, vocab_path: str):
        """
        Load vocabulary from vocab.jsonl file and initialize components.
        
        Args:
            vocab_path: Path to vocabulary file from Section 3.1 pipeline
        """
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        vocab_dict = {}
        total_prob = 0.0
        
        get_embedding_logger('vocab2embedding').info(f"Loading vocabulary from: {vocab_path}")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    # Support both 'probability' and 'prob' for backward compatibility
                    prob = entry.get('probability') or entry.get('prob')
                    
                    if 'piece' in entry and prob is not None:
                        if prob <= 0 or prob > 1:
                            get_embedding_logger('vocab2embedding').warning(f"Line {line_num}: Invalid probability {prob} for piece '{entry['piece']}'")
                            continue
                            
                        vocab_dict[entry['piece']] = prob
                        total_prob += prob
                    else:
                        missing_fields = []
                        if 'piece' not in entry:
                            missing_fields.append('piece')
                        if prob is None:
                            missing_fields.append('probability or prob')
                        get_embedding_logger('vocab2embedding').warning(f"Line {line_num}: Missing required fields: {', '.join(missing_fields)}")
                except Exception as e:
                    get_embedding_logger('vocab2embedding').error(f"Line {line_num}: Error parsing vocabulary entry: {e}")
        
        # Validate vocabulary properties
        if not vocab_dict:
            raise ValueError("No valid vocabulary entries found")
        
        get_embedding_logger('vocab2embedding').info(f"Loaded vocabulary: {len(vocab_dict)} pieces, total prob: {total_prob:.6f}")
        
        # Check for single codepoints
        single_chars = {piece for piece in vocab_dict if len(piece) == 1}
        get_embedding_logger('vocab2embedding').info(f"Single codepoint coverage: {len(single_chars)} unique characters")
        
        # Initialize pipeline components (span width will be set dynamically per sequence)
        self.unigram_lm = UnigramLM(vocab_dict, self.device)
        self.seed_embedder = SeedEmbedder(vocab_dict, self.embed_dim, self.device)
        # Initialize convolutional encoder with validated parameters
        # Get convolution configuration from nested structure (validated in _validate_convolution_config)
        architecture_config = self.config['architecture']  # Safe to access directly after validation
        conv_kernels = architecture_config['conv_kernels']
        conv_dilations = architecture_config['conv_dilations']
        
        self.conv_encoder = ConvEncoderKernel(
            self.embed_dim,
            conv_kernels,
            conv_dilations,
            self.dropout_rate, 
            self.device
        )
        
        # Store vocab_dict for dynamic candidate generator creation
        self.vocab_dict = vocab_dict
        
        # Dynamic w_max will be computed when corpus is processed
        # Set default to sequence-based bound for now
        self.w_max = self.w_max_bound  # max_sequence_length // 2
        
        get_embedding_logger('vocab2embedding').info(
            f"Dynamic w_max initialized: {self.w_max} (max_sequence_length // 2)"
        )
        
        # Initialize default candidate generator for testing/inspection purposes
        span_config = self.config.get('span_generation', {})
        self.candidate_generator = SpanCandidateGenerator(
            vocab_dict,
            tau_vocab=float(span_config.get('tau_vocab', 1e-4)),
            tau_comp=float(span_config.get('tau_comp', 1e-6)),
            w_max=self.w_max  # Use dynamic value instead of config
        )
    
    def compute_dynamic_w_max(self, sequences: List[str]) -> int:
        """
        Compute dynamic w_max based on the actual input corpus sequences.
        
        Following corpus-adaptive approach: w_max = min(longest_word_length, max_sequence_length // 2)
        where longest_word_length is found by analyzing the corpus for the longest complete word.
        This ensures span generation is adapted to actual corpus content while respecting sequence limits.
        
        Args:
            sequences: List of input sequences from the corpus
            
        Returns:
            Computed w_max value (smaller of corpus-based and sequence-based bounds)
        """
        import re
        
        max_word_length = 0
        
        get_embedding_logger('vocab2embedding').info(f"Computing dynamic w_max from {len(sequences)} sequences...")
        
        for sequence in sequences:
            if not sequence or not sequence.strip():
                continue
                
            # Split by whitespace to get words - use regex to handle multiple spaces/tabs/newlines
            words = re.split(r'\s+', sequence.strip())
            
            for word in words:
                if word:  # Skip empty strings
                    word_length = len(word)
                    if word_length > max_word_length:
                        max_word_length = word_length
        
        # Dynamic w_max: smaller of longest word or sequence-based bound for corpus adaptation
        corpus_based_w_max = max_word_length
        sequence_based_w_max = self.w_max_bound  # max_sequence_length // 2
        
        # Use the smaller value for better corpus adaptation while respecting sequence limits
        computed_w_max = min(corpus_based_w_max, sequence_based_w_max)
        
        get_embedding_logger('vocab2embedding').info(
            f"Dynamic w_max computed: {computed_w_max} "
            f"(corpus-based: {corpus_based_w_max}, sequence-based: {sequence_based_w_max}, "
            f"longest word: {max_word_length} chars)"
        )
        
        return computed_w_max
    
    def process_sequence(self, sequence: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Process a single sequence through the complete pipeline following
        Algorithm 4 from Section 3.2.5 exactly.
        
        Implementation of the Unified Seed Embedding and Candidate Generation
        algorithm with all four steps:
        1. Soft probability computation via forward-backward algorithm
        2. Seed embeddings: H^0 = P · W_emb
        3. Multi-scale contextualization: H = ConvEncoder(H^0)
        4. Vocabulary-informed candidate generation with dynamic span width
        
        Args:
            sequence: Input codepoint sequence
            metadata: Optional metadata containing sequence information
            
        Returns:
            Dictionary containing embeddings H, candidates C, probabilities P,
            and span width information exactly as specified in Algorithm 4
        """
        if (self.unigram_lm is None or self.seed_embedder is None or 
            self.conv_encoder is None or not hasattr(self, 'vocab_dict')):
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")
        
        # Use dynamic w_max computed from vocabulary structure
        span_width = self.w_max
        
        # Create candidate generator with dynamic span width
        span_config = self.config.get('span_generation', {})
        candidate_generator = SpanCandidateGenerator(
            self.vocab_dict,
            tau_vocab=float(span_config.get('tau_vocab', 1e-4)),
            tau_comp=float(span_config.get('tau_comp', 1e-6)),
            w_max=span_width
        )
        
        # Step 1: Soft Probability Computation (Equations 2-4)
        soft_probs = self.unigram_lm.forward_backward(sequence)
        
        # Step 2: Seed embeddings: H^0 = P · W_emb (Equation 5)
        seed_embeddings = self.seed_embedder(soft_probs)
        
        # Step 3: Multi-scale contextualization: H = ConvEncoder(H^0) (Equations 7-8)
        contextual_embeddings = self.conv_encoder(seed_embeddings)
        
        # Step 4: Candidate generation using Algorithm 3 (Equations 9-11)
        candidates = candidate_generator.generate_candidates(sequence)
        
        result = {
            'soft_probabilities': soft_probs.detach().cpu().numpy(),
            'seed_embeddings': seed_embeddings.detach().cpu().numpy(), 
            'contextual_embeddings': contextual_embeddings.detach().cpu().numpy(),
            'span_candidates': candidates,
            'sequence_length': len(sequence),
            'num_candidates': len(candidates),
            'span_width': span_width
        }
        
        # Add analysis if enabled
        if self.add_analysis:
            try:
                from x_spanformer.embedding.embedding_utils import analyze_embedding_quality
                result['analysis'] = analyze_embedding_quality(
                    contextual_embeddings.detach().cpu().numpy()
                )
                get_embedding_logger('vocab2embedding').debug("Added embedding quality analysis to results")
            except Exception as e:
                get_embedding_logger('vocab2embedding').warning(f"Error during embedding analysis: {e}")
        
        return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="X-Spanformer Vocab2Embedding Pipeline - Section 3.2 Implementation"
    )
    
    parser.add_argument(
        "--vocab", "-v",
        type=str, 
        required=True,
        help="Path to vocab.jsonl file from Section 3.1 pipeline"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str, 
        required=True,
        help="Path to input dataset.jsonl file with PretrainRecord format (contains 'raw' field)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True, 
        help="Output directory for embedding files"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/pipelines/vocab2embedding.yaml",
        help="Path to configuration YAML file"
    )
    
    return parser.parse_args()


def load_existing_records(output_dir: Path, pipeline_config: Dict, force: bool = False) -> Tuple[Dict[int, Dict], int]:
    """
    Load existing embedding records following the standard pattern from other pipelines.
    
    Args:
        output_dir: Output directory containing existing results
        pipeline_config: Pipeline configuration to determine which files to check
        force: If True, ignore existing records and start fresh
        
    Returns:
        Tuple of (existing_records_dict, last_processed_id)
    """
    existing_records = {}
    last_processed = 0
    
    if force:
        get_embedding_logger('vocab2embedding').info("Force mode enabled - starting fresh processing")
        return existing_records, last_processed
    
    json_dir = output_dir / "json"
    seed_dir = output_dir / "seed"
    context_dir = output_dir / "context"
    soft_prob_dir = output_dir / "soft_prob"
    
    # Context embeddings are always saved and required, so check context directory first
    if not context_dir.exists():
        get_embedding_logger('vocab2embedding').info("No existing records found - starting fresh processing")
        return existing_records, last_processed
    
    context_files = list(context_dir.glob("context_emb_*.npy"))
    if not context_files:
        get_embedding_logger('vocab2embedding').info("No existing embedding files found - starting fresh processing")
        return existing_records, last_processed
    
    get_embedding_logger('vocab2embedding').info(f"Found {len(context_files)} existing context embedding files - verifying integrity")
    
    # Load and verify existing records based on context files
    for context_file in context_files:
        try:
            # Extract sequence ID from filename pattern: context_emb_XXXXXX.npy
            seq_id = int(context_file.stem.split('_')[2])  # context_emb_XXXXXX -> XXXXXX
            
            # Verify all required files exist based on configuration
            if not verify_processed_sequence(json_dir, seed_dir, context_dir, soft_prob_dir, seq_id, pipeline_config):
                get_embedding_logger('vocab2embedding').warning(f"Sequence {seq_id} incomplete - will be reprocessed")
                continue
            
            # Load metadata if JSON is enabled, otherwise create minimal record
            output_config = pipeline_config.get('output', {})
            if output_config.get('save_json_metadata', False):
                json_file = json_dir / f"embedding_{seq_id:06d}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        existing_records[seq_id] = {
                            'sequence': data.get('sequence', ''),
                            'sequence_length': data.get('sequence_length', 0),
                            'num_candidates': data.get('num_candidates', 0),
                            'span_candidates': data.get('span_candidates', [])
                        }
            else:
                # JSON disabled - create minimal record based on context file existence
                existing_records[seq_id] = {
                    'sequence': f'[context_only_seq_{seq_id}]',  # Placeholder
                    'sequence_length': 0,  # Unknown without JSON
                    'num_candidates': 0,   # Unknown without JSON
                    'span_candidates': []  # Unknown without JSON
                }
            
            last_processed = max(last_processed, seq_id)
                
        except (ValueError, IndexError) as e:
            get_embedding_logger('vocab2embedding').warning(f"Error reading {context_file}: {e}")
            continue
        except Exception as e:
            get_embedding_logger('vocab2embedding').error(f"Unexpected error with {context_file}: {e}")
            continue
    
    if existing_records:
        get_embedding_logger('vocab2embedding').info(f"Successfully loaded {len(existing_records)} existing records")
        get_embedding_logger('vocab2embedding').info(f"Last processed sequence ID: {last_processed}")
    else:
        get_embedding_logger('vocab2embedding').info("No valid existing records found - starting fresh processing")
    
    return existing_records, last_processed

def verify_processed_sequence(json_dir: Path, seed_dir: Path, context_dir: Path, 
                             soft_prob_dir: Path, seq_id: int, pipeline_config: Dict) -> bool:
    """
    Verify that a sequence was completely processed by checking expected files based on configuration.
    
    Args:
        json_dir: Directory for JSON metadata files
        seed_dir: Directory for seed embedding files  
        context_dir: Directory for context embedding files
        soft_prob_dir: Directory for soft probability files
        seq_id: Sequence ID to check
        pipeline_config: Pipeline configuration to determine which files should exist
        
    Returns:
        bool: True if all expected files exist and are valid
    """
    try:
        files_to_check = []
        
        # Context embeddings are ALWAYS required (essential for downstream tasks)
        context_file = context_dir / f"context_emb_{seq_id:06d}.npy"
        files_to_check.append(context_file)
        
        # Check optional files based on configuration
        output_config = pipeline_config.get('output', {})
        
        if output_config.get('save_json_metadata', False):
            json_file = json_dir / f"embedding_{seq_id:06d}.json"
            files_to_check.append(json_file)
            
            # If JSON exists, verify its content
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not all(key in data for key in ['sequence_id', 'sequence', 'num_candidates']):
                        return False
        
        if output_config.get('save_seed_embeddings', False):
            seed_file = seed_dir / f"seed_emb_{seq_id:06d}.npy"
            files_to_check.append(seed_file)
        
        if output_config.get('save_soft_probabilities', False):
            # Check for soft probabilities (either .npy or .npz format)
            soft_prob_npy = soft_prob_dir / f"soft_probs_{seq_id:06d}.npy"
            soft_prob_npz = soft_prob_dir / f"soft_probs_{seq_id:06d}.npz"
            
            if soft_prob_npy.exists():
                files_to_check.append(soft_prob_npy)
            elif soft_prob_npz.exists():
                files_to_check.append(soft_prob_npz)
            else:
                return False  # Soft probs enabled but file missing
        
        # Verify all required files exist and are non-empty
        for file_path in files_to_check:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
        
        return True
        
    except Exception:
        return False

def main():
    """Main pipeline execution with graceful shutdown and resume capability."""
    global SHUTDOWN_REQUESTED
    
    # Setup signal handlers first
    setup_signal_handlers()
    
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging in the output directory (root level) - before pipeline init
    setup_embedding_logging(output_path, 'vocab2embedding')
    global logger
    logger = get_embedding_logger('vocab2embedding')

    # Log command line arguments first
    logger.info("COMMAND LINE ARGUMENTS:")
    logger.info(f"  Vocabulary file: {args.vocab}")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Config file: {args.config}")
    logger.info("-" * 80)
    
    logger.info("[bold cyan]X-Spanformer VOCAB2EMBEDDING Pipeline[/bold cyan]")
    logger.info("[green]Initializing embedding generation pipeline[/green]")
    
    # Initialize pipeline and log full configuration FIRST
    logger.info("=" * 50)
    logger.info("STAGE 1: PIPELINE INITIALIZATION")
    logger.info("=" * 50)
    
    pipeline = Vocab2EmbeddingPipeline(args.config)
    
    # Now load existing records with pipeline config
    existing_records, last_processed = load_existing_records(output_path, pipeline.config)
    if existing_records:
        logger.info(f"RESUMING PROCESSING - Found {len(existing_records)} previously processed sequences")
        logger.info(f"Last processed sequence ID: {last_processed}")
    else:
        logger.info("STARTING FRESH PROCESSING - No existing sequences found")
    
    # Log complete configuration pretty-printed
    logger.info("PIPELINE CONFIGURATION:")
    config_str = yaml.dump(pipeline.config, default_flow_style=False, indent=2, sort_keys=False)
    for line in config_str.split('\n'):
        if line.strip():
            logger.info(f"  {line}")
    logger.info(f"  Selected device: {pipeline.device}")
    logger.info("-" * 50)
    
    # Log available GPU devices in simple list format
    logger.info("DEVICE INFORMATION:")
    if torch.cuda.is_available():
        logger.info("  CUDA Devices:")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"    - {i}: {device_name} ({device_memory:.1f}GB)")
        logger.info(f"  Selected: Device {pipeline.config.get('processing', {}).get('device_id', 0)}")
    else:
        logger.info("  CUDA: Not available - using CPU")
    
    logger.info(f"Pipeline initialized on: {pipeline.device}")
    logger.info("-" * 50)

    # Create subdirectories based on configuration
    json_dir = output_path / "json"
    seed_dir = output_path / "seed" 
    context_dir = output_path / "context"
    soft_prob_dir = output_path / "soft_prob"

    # Always create contextual embeddings directory (essential for downstream tasks)
    context_dir.mkdir(exist_ok=True)
    
    # Conditionally create directories based on config
    if pipeline.config.get('output', {}).get('save_seed_embeddings', False):
        seed_dir.mkdir(exist_ok=True)
    
    if pipeline.config.get('output', {}).get('save_json_metadata', False):
        json_dir.mkdir(exist_ok=True)
    
    if pipeline.config.get('output', {}).get('save_soft_probabilities', False):
        soft_prob_dir.mkdir(exist_ok=True)
    
    # Load vocabulary
    logger.info("=" * 50)
    logger.info("STAGE 2: VOCABULARY LOADING")
    logger.info("=" * 50)
    
    pipeline.load_vocabulary(args.vocab)
    logger.info(f"Vocabulary loading completed - span width: {pipeline.config.get('w_max', 64)}")
    
    # Load sequences using shared utility
    logger.info("=" * 50)
    logger.info("STAGE 3: SEQUENCE LOADING")
    logger.info("=" * 50)
    
    logger.info(f"Loading sequences from: {args.input}")
    sequences, stats = load_pretrain_records(args.input, pipeline.max_sequence_length)
    
    if not sequences:
        logger.error("No valid sequences found in input file")
        return
    
    logger.info(f"Loaded {len(sequences)} sequences for processing")
    logger.info(f"Average sequence length: {stats.get('avg_length', 'N/A')}")
    logger.info(f"Max sequence length: {stats.get('max_length', 'N/A')}")
    logger.info(f"Min sequence length: {stats.get('min_length', 'N/A')}")
    
    # Compute dynamic w_max based on actual corpus content
    logger.info("=" * 50)
    logger.info("STAGE 4: DYNAMIC W_MAX COMPUTATION")
    logger.info("=" * 50)
    
    dynamic_w_max = pipeline.compute_dynamic_w_max(sequences)
    pipeline.w_max = dynamic_w_max
    
    # Recreate candidate generator with updated w_max
    span_config = pipeline.config.get('span_generation', {})
    pipeline.candidate_generator = SpanCandidateGenerator(
        pipeline.vocab_dict,
        tau_vocab=float(span_config.get('tau_vocab', 1e-4)),
        tau_comp=float(span_config.get('tau_comp', 1e-6)),
        w_max=dynamic_w_max
    )
    
    # Determine processing plan
    if existing_records:
        logger.info(f"RESUMING from sequence {last_processed + 1} (skipping {len(existing_records)} already processed)")
    
    # Process sequences from loaded data
    logger.info("=" * 50)
    logger.info("STAGE 5: SEQUENCE PROCESSING")
    logger.info("=" * 50)
    
    processed_count = 0
    error_count = 0
    skipped_count = len(existing_records)  # Count previously processed as skipped
    
    logger.info(f"Processing {len(sequences)} total sequences...")
    
    for seq_id, sequence in enumerate(sequences, 1):
        # Check for shutdown signal
        if SHUTDOWN_REQUESTED:
            logger.warning("SHUTDOWN SIGNAL RECEIVED - Stopping processing after current sequence")
            break
        
        # Skip already processed sequences (following other pipelines pattern)
        if seq_id in existing_records:
            continue
        
        try:
            logger.info(f"Sequence {seq_id}/{len(sequences)} - Length: {len(sequence)} chars")
            
            # GPU memory status for monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Process sequence using the pipeline (handles modality detection and span width automatically)
            result = pipeline.process_sequence(sequence)
            
            logger.info(f"  Forward-backward: {result['soft_probabilities'].shape}")
            logger.info(f"  Seed embeddings: {result['seed_embeddings'].shape}")
            logger.info(f"  Contextual embeddings: {result['contextual_embeddings'].shape}")
            logger.info(f"  Span candidates: {result['num_candidates']} (span_width: {result['span_width']})")
            
            # Save results conditionally based on configuration
            json_size = 0
            if pipeline.config.get('output', {}).get('save_json_metadata', False):
                output_file = json_dir / f"embedding_{seq_id:06d}.json"
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = {
                        'sequence_id': seq_id,
                        'sequence': sequence,
                        'sequence_length': result['sequence_length'],
                        'num_candidates': result['num_candidates'],
                        'span_candidates': result['span_candidates'],
                        'soft_probabilities_shape': result['soft_probabilities'].shape,
                        'seed_embeddings_shape': result['seed_embeddings'].shape, 
                        'contextual_embeddings_shape': result['contextual_embeddings'].shape
                    }
                    json.dump(json_result, outfile, ensure_ascii=False, indent=2)
                json_size = output_file.stat().st_size / 1024  # KB
            
            # Save embeddings as numpy files conditionally
            if pipeline.save_soft_probabilities:
                np.save(soft_prob_dir / f"soft_probs_{seq_id:06d}.npy", result['soft_probabilities'])
            
            if pipeline.save_seed_embeddings:
                np.save(seed_dir / f"seed_emb_{seq_id:06d}.npy", result['seed_embeddings'])
            
            # Always save contextual embeddings (essential for downstream tasks)
            np.save(context_dir / f"context_emb_{seq_id:06d}.npy", result['contextual_embeddings'])
            
            # Log file sizes for monitoring (handle optional files)
            seed_size = 0
            if pipeline.save_seed_embeddings:
                seed_size = (seed_dir / f"seed_emb_{seq_id:06d}.npy").stat().st_size / 1024  # KB
            context_size = (context_dir / f"context_emb_{seq_id:06d}.npy").stat().st_size / 1024  # KB
            
            if pipeline.save_soft_probabilities:
                soft_prob_size = (soft_prob_dir / f"soft_probs_{seq_id:06d}.npy").stat().st_size / 1024  # KB
                if json_size > 0 and seed_size > 0:
                    total_size = json_size + seed_size + context_size + soft_prob_size
                    logger.info(f"  Saved: JSON({json_size:.1f}KB) + Seed({seed_size:.1f}KB) + "
                              f"Context({context_size:.1f}KB) + SoftProb({soft_prob_size:.1f}KB) = {total_size:.1f}KB total")
                elif json_size > 0:
                    total_size = json_size + context_size + soft_prob_size
                    logger.info(f"  Saved: JSON({json_size:.1f}KB) + Context({context_size:.1f}KB) + "
                              f"SoftProb({soft_prob_size:.1f}KB) = {total_size:.1f}KB total (Seed: SKIPPED)")
                elif seed_size > 0:
                    total_size = seed_size + context_size + soft_prob_size
                    logger.info(f"  Saved: Seed({seed_size:.1f}KB) + Context({context_size:.1f}KB) + "
                              f"SoftProb({soft_prob_size:.1f}KB) = {total_size:.1f}KB total (JSON: SKIPPED)")
                else:
                    total_size = context_size + soft_prob_size
                    logger.info(f"  Saved: Context({context_size:.1f}KB) + SoftProb({soft_prob_size:.1f}KB) = "
                              f"{total_size:.1f}KB total (JSON: SKIPPED, Seed: SKIPPED)")
            else:
                if json_size > 0 and seed_size > 0:
                    total_size = json_size + seed_size + context_size
                    logger.info(f"  Saved: JSON({json_size:.1f}KB) + Seed({seed_size:.1f}KB) + "
                              f"Context({context_size:.1f}KB) = {total_size:.1f}KB total (SoftProb: SKIPPED)")
                elif json_size > 0:
                    total_size = json_size + context_size
                    logger.info(f"  Saved: JSON({json_size:.1f}KB) + Context({context_size:.1f}KB) = "
                              f"{total_size:.1f}KB total (Seed: SKIPPED, SoftProb: SKIPPED)")
                elif seed_size > 0:
                    total_size = seed_size + context_size
                    logger.info(f"  Saved: Seed({seed_size:.1f}KB) + Context({context_size:.1f}KB) = "
                              f"{total_size:.1f}KB total (JSON: SKIPPED, SoftProb: SKIPPED)")
                else:
                    total_size = context_size
                    logger.info(f"  Saved: Context({context_size:.1f}KB) = {total_size:.1f}KB total "
                              f"(JSON: SKIPPED, Seed: SKIPPED, SoftProb: SKIPPED)")
            
            processed_count += 1
            logger.info("-" * 50)
                
        except Exception as e:
            logger.error(f"Sequence {seq_id}: Error processing sequence: {e}")
            error_count += 1
            continue
    
    # Final statistics
    total_processed = skipped_count + processed_count
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 50)
    
    if SHUTDOWN_REQUESTED:
        logger.warning("Pipeline terminated by user request (graceful shutdown)")
    else:
        logger.info("Pipeline completed normally")
    
    logger.info(f"Total sequences in dataset: {len(sequences)}")
    logger.info(f"Previously processed (skipped): {skipped_count}")
    logger.info(f"Newly processed in this run: {processed_count}")
    logger.info(f"Total processed: {total_processed}")
    
    if error_count > 0:
        logger.warning(f"Errors encountered: {error_count} sequences failed")
    
    if total_processed < len(sequences):
        remaining = len(sequences) - total_processed
        logger.info(f"Remaining to process: {remaining} sequences")
        logger.info("Use the same command to resume processing from where it left off")
    
    logger.info(f"Output saved to: {output_path}")
    logger.info("Output structure:")
    
    # Only log directories that exist
    if json_dir.exists():
        logger.info(f"  JSON metadata: {json_dir} ({len(list(json_dir.glob('*.json')))} files)")
    else:
        logger.info(f"  JSON metadata: Not saved (disabled in config)")
        
    if seed_dir.exists():
        logger.info(f"  Seed embeddings: {seed_dir} ({len(list(seed_dir.glob('*.npy')))} files)")
    else:
        logger.info(f"  Seed embeddings: Not saved (disabled for performance)")
        
    logger.info(f"  Context embeddings: {context_dir} ({len(list(context_dir.glob('*.npy')))} files)")
    
    if soft_prob_dir.exists():
        logger.info(f"  Soft probabilities: {soft_prob_dir} ({len(list(soft_prob_dir.glob('*.npy')))} files)")
    else:
        logger.info(f"  Soft probabilities: Not saved (disabled for performance)")
        
    logger.info(f"  Log file: {output_path / 'embedding.log'}")
    
    # Exit code for automation
    if SHUTDOWN_REQUESTED:
        logger.info("Exiting with code 130 (interrupted)")
        sys.exit(130)
    elif error_count > 0:
        logger.info("Exiting with code 1 (errors encountered)")
        sys.exit(1)
    else:
        logger.info("Exiting with code 0 (success)")
        sys.exit(0)


if __name__ == "__main__":
    main()
