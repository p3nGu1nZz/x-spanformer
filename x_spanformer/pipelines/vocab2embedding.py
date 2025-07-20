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

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import warnings

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

# Module-level logger that gets configured in main()
logger = None

def get_logger() -> logging.Logger:
    """Get the module logger, creating a basic one if none exists."""
    global logger
    if logger is None:
        logger = get_embedding_logger('vocab2embedding')
    return logger


def load_corpus(corpus_path: str) -> List[str]:
    """
    Load sequences from a JSONL file using PretrainRecord format.
    
    This function mirrors the load_corpus() pattern from jsonl2vocab.py
    and handles only the PretrainRecord format with 'raw' field extraction.
    
    Args:
        corpus_path: Path to the JSONL corpus file with PretrainRecord format
        
    Returns:
        List of text sequences extracted from the corpus
        
    Raises:
        FileNotFoundError: If the corpus file doesn't exist
        ValueError: If no valid sequences are found
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("STAGE 1: CORPUS LOADING")
    logger.info("=" * 50)
    
    sequences = []
    total_records = 0
    valid_records = 0
    invalid_records = 0
    discarded_records = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                total_records += 1
                record = json.loads(line)
                
                # Only support PretrainRecord format with 'raw' field
                if isinstance(record, dict) and 'raw' in record:
                    sequence = record['raw']
                    
                    # Skip discarded sequences
                    if 'meta' in record and record['meta'].get('status') == 'discard':
                        logger.debug(f"Line {line_num}: Skipping discarded sequence")
                        discarded_records += 1
                        continue
                    
                    if sequence and sequence.strip():
                        sequences.append(sequence.strip())
                        valid_records += 1
                    else:
                        logger.warning(f"Line {line_num}: Empty 'raw' field")
                        invalid_records += 1
                else:
                    logger.warning(f"Line {line_num}: Record missing 'raw' field - only PretrainRecord format supported")
                    invalid_records += 1
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                invalid_records += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                invalid_records += 1
    
    # Calculate corpus statistics
    total_chars = sum(len(seq) for seq in sequences)
    avg_sequence_length = total_chars / len(sequences) if sequences else 0
    
    logger.info("-" * 50)
    logger.info("CORPUS LOADING SUMMARY:")
    logger.info(f"  Total input records: {total_records}")
    logger.info(f"  Valid sequences: {valid_records}")
    logger.info(f"  Discarded sequences: {discarded_records}")
    logger.info(f"  Invalid records: {invalid_records}")
    if total_records > 0:
        logger.info(f"  Success rate: {valid_records/total_records*100:.1f}%")
    else:
        logger.info(f"  Success rate: N/A (no records)")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average sequence length: {avg_sequence_length:.1f} chars")
    logger.info("-" * 50)
    
    if not sequences:
        raise ValueError(f"No valid sequences found in corpus file: {corpus_path}")
    
    return sequences


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
        Compute soft piece probabilities using forward-backward algorithm.
        
        Args:
            sequence: Input codepoint sequence as string
            
        Returns:
            Probability matrix P ∈ R^{T × |V|} where P[t,i] is the probability
            of piece u_i starting at position t
        """
        T = len(sequence)
        V = self.vocab_size
        
        # Initialize forward probabilities α_t
        # α_t = probability of generating sequence[0:t]
        alpha = torch.full((T + 1,), -float('inf'), device=self.device)
        alpha[0] = 0.0  # log(1) = 0
        
        # Forward pass
        for t in range(T):
            if alpha[t] == -float('inf'):
                continue
                
            for piece_idx, piece in self.idx_to_piece.items():
                if self.matches_at_position(sequence, t, piece):
                    next_pos = t + len(piece)
                    if next_pos <= T:
                        # α_{t+|u_i|} += α_t * p(u_i)  (in log space)
                        alpha[next_pos] = torch.logsumexp(
                            torch.stack([alpha[next_pos], 
                                       alpha[t] + self.log_piece_probs[piece_idx]]),
                            dim=0
                        )
        
        # Initialize backward probabilities β_t
        # β_t = probability of generating sequence[t:T] 
        beta = torch.full((T + 1,), -float('inf'), device=self.device)
        beta[T] = 0.0  # log(1) = 0
        
        # Backward pass
        for t in range(T - 1, -1, -1):
            for piece_idx, piece in self.idx_to_piece.items():
                if self.matches_at_position(sequence, t, piece):
                    next_pos = t + len(piece)
                    if next_pos <= T and beta[next_pos] != -float('inf'):
                        # β_t += p(u_i) * β_{t+|u_i|}  (in log space)
                        beta[t] = torch.logsumexp(
                            torch.stack([beta[t],
                                       self.log_piece_probs[piece_idx] + beta[next_pos]]),
                            dim=0
                        )
        
        # Compute soft piece probabilities
        # P[t,i] = α_t * p(u_i) * β_{t+|u_i|} / α_T
        P = torch.zeros((T, V), device=self.device)
        
        normalization = alpha[T]  # Total probability of sequence
        if normalization == -float('inf'):
            get_logger().warning(f"Sequence cannot be segmented with given vocabulary")
            return P
        
        for t in range(T):
            for piece_idx, piece in self.idx_to_piece.items():
                if self.matches_at_position(sequence, t, piece):
                    next_pos = t + len(piece)
                    if next_pos <= T:
                        log_prob = (alpha[t] + 
                                  self.log_piece_probs[piece_idx] + 
                                  beta[next_pos] - 
                                  normalization)
                        P[t, piece_idx] = torch.exp(log_prob)
        
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
        
        # Initialize embeddings using Eq. (4) from Section 3.2.2
        pieces = list(vocab_dict.keys())
        probs = list(vocab_dict.values())
        
        for i, (piece, prob) in enumerate(zip(pieces, probs)):
            if len(piece) == 1:  # Single codepoint - standard Xavier
                std = math.sqrt(2.0 / embed_dim)
            else:  # Multi-codepoint piece - probability-adjusted Xavier
                std = math.sqrt(2.0 / (embed_dim * prob))
            
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
        
        # Multi-scale dilated convolutions with kernel sizes [3,5,7] and dilations [1,2,4]
        # Padding calculation: (kernel_size - 1) * dilation // 2
        self.conv_layers = nn.ModuleList([
            # Scale 1: kernel=3, dilation=1, padding=(3-1)*1//2=1
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, dilation=1, padding=1),
            # Scale 2: kernel=5, dilation=2, padding=(5-1)*2//2=4  
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, dilation=2, padding=4),
            # Scale 3: kernel=7, dilation=4, padding=(7-1)*4//2=12
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, dilation=4, padding=12)
        ])
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.output_proj = nn.Linear(3 * embed_dim, embed_dim)
        
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
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.gelu(conv(x))
            # Ensure output length matches input length by trimming if necessary
            if conv_out.size(-1) != x.size(-1):
                # Trim to match input length
                diff = conv_out.size(-1) - x.size(-1)
                conv_out = conv_out[..., diff//2:-(diff-diff//2)] if diff > 0 else conv_out
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features: list of (d, T) -> (3d, T)
        concatenated = torch.cat(conv_outputs, dim=-2)  # Concatenate along channel dimension
        
        # Transpose back and project: (3d, T) -> (T, 3d) -> (T, d)
        concatenated = concatenated.transpose(-2, -1)
        output = self.output_proj(concatenated)
        
        # Residual connection and layer norm
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
    
    def vocabulary_alignment(self, span_text: str) -> bool:
        """Check if span has high-probability vocabulary alignment."""
        if span_text in self.vocab_dict:
            prob = self.vocab_dict[span_text]
            if not isinstance(prob, (int, float)):
                get_logger().warning(f"Non-numeric probability for '{span_text}': {prob} (type: {type(prob)})")
                return False
            
            # Ensure tau_vocab is also numeric
            tau = self.tau_vocab
            if not isinstance(tau, (int, float)):
                get_logger().warning(f"Non-numeric tau_vocab: {tau} (type: {type(tau)})")
                return False
                
            return prob >= tau
        return False
    
    def compositional_potential(self, span_text: str) -> bool:
        """Check if span has compositional segmentation potential."""
        # Simple greedy segmentation to estimate compositional probability
        pos = 0
        log_prob = 0.0
        
        while pos < len(span_text):
            # Find longest matching piece
            best_piece = None
            best_len = 0
            
            for piece in self.vocab_dict:
                if (len(piece) > best_len and 
                    pos + len(piece) <= len(span_text) and
                    span_text[pos:pos + len(piece)] == piece):
                    best_piece = piece
                    best_len = len(piece)
            
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
    
    def generate_candidates(self, sequence: str) -> List[Tuple[int, int]]:
        """
        Generate filtered span candidates using Algorithm 3.
        
        Args:
            sequence: Input codepoint sequence
            
        Returns:
            List of (start, end) candidate span positions
        """
        T = len(sequence)
        candidates = []
        
        for i in range(T - 1):
            for j in range(i + 1, min(i + self.w_max + 1, T + 1)):
                span_text = sequence[i:j]
                
                # Apply filtering criteria from Algorithm 3
                if (self.vocabulary_alignment(span_text) or
                    self.compositional_potential(span_text) or  
                    self.whitespace_coherent(span_text)):
                    candidates.append((i, j))
        
        return candidates


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
        
        # Initialize components (will be set when vocabulary is loaded)
        self.unigram_lm: Optional[UnigramLM] = None
        self.seed_embedder: Optional[SeedEmbedder] = None  
        self.conv_encoder: Optional[ConvEncoder] = None
        self.candidate_generator: Optional[SpanCandidateGenerator] = None
        
        get_logger().info(f"Initialized vocab2embedding pipeline on {device}")
    
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
        vocab_dict = {}
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if 'piece' in entry and ('probability' in entry or 'prob' in entry):
                    # Support both 'probability' and 'prob' field names
                    prob = entry.get('probability', entry.get('prob'))
                    if isinstance(prob, str):
                        prob = float(prob)
                    vocab_dict[entry['piece']] = prob
        
        get_logger().info(f"Loaded vocabulary with {len(vocab_dict)} pieces")
        
        # Initialize pipeline components
        embed_dim = self.config.get('embed_dim', 256)
        
        self.unigram_lm = UnigramLM(vocab_dict, self.device)
        self.seed_embedder = SeedEmbedder(vocab_dict, embed_dim, self.device).to(self.device)
        self.conv_encoder = ConvEncoder(embed_dim, self.device).to(self.device)
        self.candidate_generator = SpanCandidateGenerator(
            vocab_dict,
            tau_vocab=float(self.config.get('tau_vocab', 1e-4)),
            tau_comp=float(self.config.get('tau_comp', 1e-6)),
            w_max=int(self.config.get('w_max', 64))
        )
    
    def process_sequence(self, sequence: str) -> Dict:
        """
        Process a single sequence through the complete pipeline.
        
        Args:
            sequence: Input codepoint sequence
            
        Returns:
            Dictionary containing embeddings, candidates, and probabilities
        """
        if (self.unigram_lm is None or self.seed_embedder is None or 
            self.conv_encoder is None or self.candidate_generator is None):
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")
        
        # Step 1: Soft probability computation
        soft_probs = self.unigram_lm.forward_backward(sequence)
        
        # Step 2: Seed embeddings  
        seed_embeddings = self.seed_embedder(soft_probs)
        
        # Step 3: Contextualization
        contextual_embeddings = self.conv_encoder(seed_embeddings)
        
        # Step 4: Candidate generation
        candidates = self.candidate_generator.generate_candidates(sequence)
        
        return {
            'soft_probabilities': soft_probs.detach().cpu().numpy(),
            'seed_embeddings': seed_embeddings.detach().cpu().numpy(), 
            'contextual_embeddings': contextual_embeddings.detach().cpu().numpy(),
            'span_candidates': candidates,
            'sequence_length': len(sequence),
            'num_candidates': len(candidates)
        }


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
    
    # Setup logging in the output directory
    setup_embedding_logging(output_path, 'vocab2embedding')
    global logger
    logger = get_embedding_logger('vocab2embedding')
    
    logger.info("Starting vocab2embedding pipeline")
    logger.info(f"Vocabulary: {args.vocab}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Initialize pipeline
    pipeline = Vocab2EmbeddingPipeline(args.config, args.device)
    pipeline.load_vocabulary(args.vocab)
    
    # Process sequences from file
    processed_count = 0
    start_time = time.time()

    logger.info(f"Processing sequences from: {args.input}")
    
    with open(args.input, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line)
                
                # Extract sequence using PretrainRecord format only
                sequence = None
                if isinstance(record, dict) and 'raw' in record:
                    sequence = record['raw']
                    # Skip discarded sequences
                    if 'meta' in record and record['meta'].get('status') == 'discard':
                        logger.debug(f"Line {line_num}: Skipping discarded sequence")
                        continue
                else:
                    logger.warning(f"Line {line_num}: Record missing 'raw' field - only PretrainRecord format supported")
                    continue

                # Validate sequence
                if not sequence or not sequence.strip():
                    logger.warning(f"Line {line_num}: Empty sequence")
                    continue
                
                # Skip sequences that are too long
                if len(sequence) > args.max_length:
                    logger.warning(f"Line {line_num}: Sequence too long ({len(sequence)} > {args.max_length})")
                    continue
                
                # Process sequence
                result = pipeline.process_sequence(sequence)
                
                # Save results
                output_file = output_path / f"embedding_{line_num:06d}.json"
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = {
                        'sequence_id': line_num,
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
                np.save(output_path / f"soft_probs_{line_num:06d}.npy", result['soft_probabilities'])
                np.save(output_path / f"seed_emb_{line_num:06d}.npy", result['seed_embeddings'])
                np.save(output_path / f"context_emb_{line_num:06d}.npy", result['contextual_embeddings'])
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    logger.info(f"Processed {processed_count} sequences ({rate:.2f} seq/sec)")
                    
            except Exception as e:
                logger.error(f"Line {line_num}: Error processing sequence: {e}")
                continue
    
    # Final statistics
    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed: {processed_count} sequences in {elapsed:.2f} seconds")
    logger.info(f"Average processing rate: {processed_count / elapsed:.2f} sequences/second")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
