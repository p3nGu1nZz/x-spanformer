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
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.vocab import VocabStats
from x_spanformer.embedding.embedding_logging import setup_embedding_logging, get_embedding_logger
from x_spanformer.embedding.embedding_utils import (
    analyze_embedding_quality
)
from x_spanformer.pipelines.shared.jsonl_processor import load_pretrain_records

# Module-level logger that gets configured in main()
logger = None


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
        
        # Build prefix tree for efficient matching
        self._build_prefix_tree(vocab_dict.keys())
    
    def _build_prefix_tree(self, pieces):
        """Build prefix tree for efficient piece matching."""
        self.prefix_tree = {}
        for piece in pieces:
            node = self.prefix_tree
            for char in piece:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['_end'] = self.piece_to_idx[piece]
    
    def _find_matches_at_position(self, sequence: str, pos: int):
        """Find all pieces that match starting at position pos using prefix tree."""
        matches = []
        node = self.prefix_tree
        
        for i in range(pos, min(pos + self.max_piece_length, len(sequence))):
            char = sequence[i]
            if char not in node:
                break
            node = node[char]
            if '_end' in node:
                piece_idx = node['_end']
                piece_len = i - pos + 1
                matches.append((piece_idx, piece_len))
        
        return matches
    
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
                
            # Find all pieces that match starting at position t
            matches = self._find_matches_at_position(sequence, t)
            for piece_idx, piece_len in matches:
                next_pos = t + piece_len
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
            matches = self._find_matches_at_position(sequence, t)
            for piece_idx, piece_len in matches:
                next_pos = t + piece_len
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
            matches = self._find_matches_at_position(sequence, t)
            for piece_idx, piece_len in matches:
                next_pos = t + piece_len
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


class ConvEncoder(nn.Module):
    """
    Multi-scale dilated convolutional encoder implementing the contextual
    embedding computation described in Section 3.2.3.
    """
    
    def __init__(self, embed_dim: int, device: str = 'cuda'):
        """
        Initialize multi-scale convolutional encoder.
        
        Args:
            embed_dim: Embedding dimension d
            device: PyTorch device for computation
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        
        # Multi-scale dilated convolutions following Section 3.2.3 specifications
        # K = {3, 5, 7} kernel sizes and D = {1, 2, 4} dilation rates
        # Creates |K| × |D| = 9 distinct receptive field patterns
        self.conv_layers = nn.ModuleList([
            # Kernel size 3 with all dilations
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=1, padding=1),   # RF = 3
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=2, padding=2),   # RF = 5
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=4, padding=4),   # RF = 9
            # Kernel size 5 with all dilations  
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, dilation=1, padding=2),   # RF = 5
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, dilation=2, padding=4),   # RF = 9
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, dilation=4, padding=8),   # RF = 17
            # Kernel size 7 with all dilations
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, dilation=1, padding=3),   # RF = 7
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, dilation=2, padding=6),   # RF = 13
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, dilation=4, padding=12),  # RF = 25
        ])
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Output projection from 9d back to d (not 3d)
        self.output_proj = nn.Linear(9 * embed_dim, embed_dim)
        
        # Move all modules to the specified device
        self.to(device)
    
    def forward(self, seed_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale contextualization to seed embeddings.
        
        Args:
            seed_embeddings: Seed embeddings H^0 ∈ R^{T × d}
            
        Returns:
            Contextual embeddings H ∈ R^{T × d}
        """
        # Transpose for conv1d: (T, d) -> (d, T)
        x = seed_embeddings.transpose(-2, -1)
        
        # Apply multi-scale convolutions following Equation (7)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.gelu(conv(x))
            # Ensure output length matches input length by trimming if necessary
            if conv_out.size(-1) != x.size(-1):
                # Trim to match input length
                diff = conv_out.size(-1) - x.size(-1)
                conv_out = conv_out[..., diff//2:-(diff-diff//2)] if diff > 0 else conv_out
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features: list of (d, T) -> (9d, T)
        concatenated = torch.cat(conv_outputs, dim=-2)  # Concatenate along channel dimension
        
        # Transpose back and project: (9d, T) -> (T, 9d) -> (T, d)
        concatenated = concatenated.transpose(-2, -1)
        output = self.output_proj(concatenated)
        
        # Residual connection and layer norm following Equation (8)
        output = self.layer_norm(output + seed_embeddings)
        output = self.dropout(output)
        
        return output


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
        """Check if span respects whitespace boundaries."""
        # Simple heuristic: span should not split words
        if not span_text:
            return False
        
        # Allow spans that start/end at word boundaries or are complete words
        starts_at_boundary = span_text[0].isspace() or not span_text[0].isalpha()
        ends_at_boundary = span_text[-1].isspace() or not span_text[-1].isalpha()
        stripped = span_text.strip()
        is_complete_word = bool(stripped) and not any(c.isspace() for c in stripped)
        
        return starts_at_boundary or ends_at_boundary or is_complete_word
    
    def _process_span_chunk(self, args):
        """Process a chunk of span candidates in parallel."""
        sequence, start_positions, end_limit = args
        candidates = []
        vocab_aligned = 0
        comp_potential = 0
        whitespace_coherent = 0
        
        for i in start_positions:
            for j in range(i + 1, min(i + self.w_max, end_limit) + 1):
                span_text = sequence[i:j]
                
                # Apply the three filtering criteria exactly as in paper:
                if self.vocabulary_alignment(span_text):
                    candidates.append((i, j))
                    vocab_aligned += 1
                elif self.compositional_potential(span_text):
                    candidates.append((i, j))
                    comp_potential += 1
                elif self.whitespace_coherent(span_text):
                    candidates.append((i, j))
                    whitespace_coherent += 1
        
        return candidates, vocab_aligned, comp_potential, whitespace_coherent
    
    def generate_candidates(self, sequence: str) -> List[Tuple[int, int]]:
        """
        Generate filtered span candidates exactly following Algorithm 3 from Section 3.2.4.
        Uses parallel processing for all sequences to maximize performance.
        
        Note: Paper uses 1-indexed positions, we return 0-indexed tuples but follow
        the exact algorithm logic from the paper.
        
        Args:
            sequence: Input codepoint sequence
            
        Returns:
            List of (start, end) candidate span positions (0-indexed, exclusive end)
        """
        T = len(sequence)
        
        # Use parallel processing for all sequences
        # Split work into chunks based on starting positions
        num_workers = min(cpu_count(), 4)  # Use up to 4 workers
        chunk_size = max(5, (T - 1) // num_workers)  # Ensure minimum chunk size of 5
        
        chunks = []
        for worker_id in range(num_workers):
            start_idx = worker_id * chunk_size
            end_idx = min((worker_id + 1) * chunk_size, T - 1)
            if start_idx >= T - 1:
                break
            
            start_positions = list(range(start_idx, end_idx))
            if start_positions:
                chunks.append((sequence, start_positions, T))
        
        if not chunks:
            return []
        
        # Process chunks in parallel
        with Pool(processes=min(len(chunks), num_workers)) as pool:
            results = pool.map(self._process_span_chunk, chunks)
        
        # Combine results from all chunks
        all_candidates = []
        for candidates, vocab_aligned, comp_potential, whitespace_coherent in results:
            all_candidates.extend(candidates)
        
        return all_candidates


class Vocab2EmbeddingPipeline:
    """
    Main pipeline class implementing the unified algorithm from Section 3.2.5.
    """
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        """
        Initialize the vocab2embedding pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            device: PyTorch device for computation
        """
        self.device = device
        self.config = self._load_config(config_path)
        
        # Extract all config parameters
        self.embed_dim = self.config.get('embed_dim', 256)
        self.max_sequence_length = self.config.get('max_sequence_length', 512)
        self.epsilon = self.config.get('epsilon', 1e-12)
        self.max_piece_length = self.config.get('max_piece_length', 16)
        self.save_intermediate = self.config.get('save_intermediate', True)
        self.save_numpy_arrays = self.config.get('save_numpy_arrays', True)
        self.save_json_metadata = self.config.get('save_json_metadata', True)
        self.add_analysis = self.config.get('add_analysis', False)
        
        # Initialize components (will be set when vocabulary is loaded)
        self.unigram_lm: Optional[UnigramLM] = None
        self.seed_embedder: Optional[SeedEmbedder] = None  
        self.conv_encoder: Optional[ConvEncoder] = None
        self.candidate_generator: Optional[SpanCandidateGenerator] = None
        
        get_embedding_logger('vocab2embedding').info(f"Initialized vocab2embedding pipeline on {device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
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
                    if 'piece' in entry and ('probability' in entry or 'prob' in entry):
                        # Support both 'probability' and 'prob' field names
                        prob = entry.get('probability', entry.get('prob'))
                        if isinstance(prob, str):
                            prob = float(prob)
                        
                        if prob <= 0 or prob > 1:
                            get_embedding_logger('vocab2embedding').warning(f"Line {line_num}: Invalid probability {prob} for piece '{entry['piece']}'")
                            continue
                            
                        vocab_dict[entry['piece']] = prob
                        total_prob += prob
                    else:
                        get_embedding_logger('vocab2embedding').warning(f"Line {line_num}: Missing required fields")
                except Exception as e:
                    get_embedding_logger('vocab2embedding').error(f"Line {line_num}: Error parsing vocabulary entry: {e}")
        
        # Validate vocabulary properties
        if not vocab_dict:
            raise ValueError("No valid vocabulary entries found")
        
        get_embedding_logger('vocab2embedding').info(f"Loaded vocabulary: {len(vocab_dict)} pieces, total prob: {total_prob:.6f}")
        
        # Check for single codepoints
        single_chars = {piece for piece in vocab_dict if len(piece) == 1}
        get_embedding_logger('vocab2embedding').info(f"Single codepoint coverage: {len(single_chars)} unique characters")
        
        # Initialize pipeline components
        self.unigram_lm = UnigramLM(vocab_dict, self.device)
        self.seed_embedder = SeedEmbedder(vocab_dict, self.embed_dim, self.device).to(self.device)
        self.conv_encoder = ConvEncoder(self.embed_dim, self.device).to(self.device)
        self.candidate_generator = SpanCandidateGenerator(
            vocab_dict,
            tau_vocab=float(self.config.get('tau_vocab', 1e-4)),
            tau_comp=float(self.config.get('tau_comp', 1e-6)),
            w_max=int(self.config.get('w_max', 64))
        )
    
    def process_sequence(self, sequence: str) -> Dict:
        """
        Process a single sequence through the complete pipeline following
        Algorithm 4 from Section 3.2.5 exactly.
        
        Implementation of the Unified Seed Embedding and Candidate Generation
        algorithm with all four steps:
        1. Soft probability computation via forward-backward algorithm
        2. Seed embeddings: H^0 = P · W_emb
        3. Multi-scale contextualization: H = ConvEncoder(H^0)
        4. Vocabulary-informed candidate generation
        
        Args:
            sequence: Input codepoint sequence
            
        Returns:
            Dictionary containing embeddings H, candidates C, and probabilities P
            exactly as specified in Algorithm 4
        """
        if (self.unigram_lm is None or self.seed_embedder is None or 
            self.conv_encoder is None or self.candidate_generator is None):
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")
        
        # Step 1: Soft Probability Computation (Equations 2-4)
        soft_probs = self.unigram_lm.forward_backward(sequence)
        
        # Step 2: Seed embeddings: H^0 = P · W_emb (Equation 5)
        seed_embeddings = self.seed_embedder(soft_probs)
        
        # Step 3: Multi-scale contextualization: H = ConvEncoder(H^0) (Equations 7-8)
        contextual_embeddings = self.conv_encoder(seed_embeddings)
        
        # Step 4: Candidate generation using Algorithm 3 (Equations 9-11)
        candidates = self.candidate_generator.generate_candidates(sequence)
        
        result = {
            'soft_probabilities': soft_probs.detach().cpu().numpy(),
            'seed_embeddings': seed_embeddings.detach().cpu().numpy(), 
            'contextual_embeddings': contextual_embeddings.detach().cpu().numpy(),
            'span_candidates': candidates,
            'sequence_length': len(sequence),
            'num_candidates': len(candidates)
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
        "--vocab", 
        type=str, 
        required=True,
        help="Path to vocab.jsonl file from Section 3.1 pipeline"
    )
    
    parser.add_argument(
        "--input",
        type=str, 
        required=True,
        help="Path to input dataset.jsonl file with PretrainRecord format (contains 'raw' field)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True, 
        help="Output directory for embedding files"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipelines/vocab2embedding.yaml",
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device for computation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing sequences"
    )
    
    parser.add_argument(
        "--max-length",
        type=int, 
        default=512,
        help="Maximum sequence length to process"
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different file types
    json_dir = output_path / "json"
    seed_dir = output_path / "seed"
    context_dir = output_path / "context"
    soft_prob_dir = output_path / "soft_prob"
    
    json_dir.mkdir(exist_ok=True)
    seed_dir.mkdir(exist_ok=True)
    context_dir.mkdir(exist_ok=True)
    soft_prob_dir.mkdir(exist_ok=True)
    
    # Setup logging in the output directory (root level)
    setup_embedding_logging(output_path, 'vocab2embedding')
    global logger
    logger = get_embedding_logger('vocab2embedding')
    
    # Log command line arguments
    logger.info("COMMAND LINE ARGUMENTS:")
    logger.info(f"  Vocabulary file: {args.vocab}")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info("-" * 80)
    
    logger.info("[bold cyan]X-Spanformer VOCAB2EMBEDDING Pipeline[/bold cyan]")
    logger.info("[green]Initializing embedding generation pipeline[/green]")
    
    # Initialize pipeline
    logger.info("=" * 50)
    logger.info("STAGE 1: PIPELINE INITIALIZATION")
    logger.info("=" * 50)
    
    # Log available GPU devices
    logger.info("GPU DEVICE INFORMATION:")
    if torch.cuda.is_available():
        logger.info(f"  CUDA available: True")
        logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"    Device {i}: {device_name} ({device_memory:.1f}GB)")
        
        # Use NVIDIA GPU (device 1) if available, otherwise device 0
        if torch.cuda.device_count() > 1 and args.device == "cuda":
            args.device = "cuda:1"  # Use RTX 3080 instead of integrated GPU
            logger.info(f"  Selected device: {args.device} (NVIDIA RTX 3080)")
        else:
            logger.info(f"  Selected device: {args.device}")
    else:
        logger.info(f"  CUDA available: False")
        logger.info(f"  Using CPU device")
    
    pipeline = Vocab2EmbeddingPipeline(args.config, args.device)
    logger.info(f"Pipeline initialization completed")
    
    # Load vocabulary
    logger.info("=" * 50)
    logger.info("STAGE 2: VOCABULARY LOADING")
    logger.info("=" * 50)
    
    pipeline.load_vocabulary(args.vocab)
    logger.info(f"Vocabulary loading completed")
    
    # Load sequences using shared utility
    logger.info("=" * 50)
    logger.info("STAGE 3: SEQUENCE LOADING")
    logger.info("=" * 50)
    
    logger.info(f"Loading sequences from: {args.input}")
    sequences, stats = load_pretrain_records(args.input, args.max_length)
    
    if not sequences:
        logger.error("No valid sequences found in input file")
        return
    
    logger.info(f"Loaded {len(sequences)} sequences for processing")
    logger.info(f"Average sequence length: {stats.get('avg_length', 'N/A')}")
    logger.info(f"Max sequence length: {stats.get('max_length', 'N/A')}")
    logger.info(f"Min sequence length: {stats.get('min_length', 'N/A')}")
    
    # Process sequences from loaded data
    logger.info("=" * 50)
    logger.info("STAGE 4: SEQUENCE PROCESSING")
    logger.info("=" * 50)
    
    processed_count = 0
    error_count = 0
    
    logger.info(f"Processing {len(sequences)} sequences...")
    
    for seq_id, sequence in enumerate(sequences, 1):
        try:
            logger.info(f"Sequence {seq_id}/{len(sequences)} - Length: {len(sequence)} chars")
            
            # GPU memory status for monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Verify pipeline is ready
            if (pipeline.unigram_lm is None or pipeline.seed_embedder is None or 
                pipeline.conv_encoder is None or pipeline.candidate_generator is None):
                raise RuntimeError("Pipeline components not initialized. This shouldn't happen after vocabulary loading.")
            
            # Step 1: Forward-backward algorithm
            soft_probs = pipeline.unigram_lm.forward_backward(sequence)
            logger.info(f"  Forward-backward: {soft_probs.shape}")
            
            # Step 2: Seed embeddings  
            seed_embeddings = pipeline.seed_embedder(soft_probs)
            logger.info(f"  Seed embeddings: {seed_embeddings.shape}")
            
            # Step 3: Contextualization
            contextual_embeddings = pipeline.conv_encoder(seed_embeddings)
            logger.info(f"  Contextual embeddings: {contextual_embeddings.shape}")
            
            # Step 4: Candidate generation
            candidates = pipeline.candidate_generator.generate_candidates(sequence)
            logger.info(f"  Span candidates: {len(candidates)}")
            
            # Combine results
            result = {
                'soft_probabilities': soft_probs.detach().cpu().numpy(),
                'seed_embeddings': seed_embeddings.detach().cpu().numpy(), 
                'contextual_embeddings': contextual_embeddings.detach().cpu().numpy(),
                'span_candidates': candidates,
                'sequence_length': len(sequence),
                'num_candidates': len(candidates)
            }
            
            # Save results
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
            
            # Save embeddings as numpy files for efficient loading
            np.save(soft_prob_dir / f"soft_probs_{seq_id:06d}.npy", result['soft_probabilities'])
            np.save(seed_dir / f"seed_emb_{seq_id:06d}.npy", result['seed_embeddings'])
            np.save(context_dir / f"context_emb_{seq_id:06d}.npy", result['contextual_embeddings'])
            
            processed_count += 1
            logger.info(f"  Saved: JSON + 3 numpy arrays")
            logger.info("-" * 50)
                
        except Exception as e:
            logger.error(f"Sequence {seq_id}: Error processing sequence: {e}")
            error_count += 1
            continue
    
    # Final statistics
    logger.info(f"Pipeline completed: {processed_count} sequences processed")
    if error_count > 0:
        logger.info(f"Errors: {error_count} sequences failed")
    logger.info(f"Output saved to: {output_path}")
    logger.info("Output structure:")
    logger.info(f"  JSON metadata: {json_dir}")
    logger.info(f"  Seed embeddings: {seed_dir}")
    logger.info(f"  Context embeddings: {context_dir}")
    logger.info(f"  Soft probabilities: {soft_prob_dir}")
    logger.info(f"  Log file: {output_path / 'embedding.log'}")


if __name__ == "__main__":
    main()
