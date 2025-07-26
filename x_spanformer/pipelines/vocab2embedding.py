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
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from multiprocessing import Process, Queue, Manager, current_process
from queue import Empty, Full
import threading

import numpy as np
import torch
import torch.nn as nn

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.embedding.embedding_logging import setup_embedding_logging, get_embedding_logger
from x_spanformer.embedding.embedding_utils import (
    analyze_embedding_quality
)
from x_spanformer.embedding.embedding_chunk import (
    ChunkManager, ChunkMetadata, save_sequence_individually_chunked
)
from x_spanformer.pipelines.shared.jsonl_processor import load_pretrain_records
from x_spanformer.kernel import ConvEncoderKernel, validate_convolution_parameters

# Module-level logger that gets configured in main()
logger = None
logger = None

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

class WorkerTask:
    """Data structure for worker task distribution."""
    def __init__(self, seq_id: int, sequence: str, config_path: str, vocab_path: str, dynamic_w_max: int):
        self.seq_id = seq_id
        self.sequence = sequence
        self.config_path = config_path
        self.vocab_path = vocab_path
        self.dynamic_w_max = dynamic_w_max

class ProcessingResult:
    """Data structure for processing results."""
    def __init__(self, seq_id: int, success: bool, result: Optional[Dict] = None, error: Optional[str] = None):
        self.seq_id = seq_id
        self.success = success
        self.result = result
        self.error = error

def sequence_processor_worker(task_queue: Queue, result_queue: Queue, worker_id: int):
    """
    Worker process function for parallel sequence processing.
    
    Each worker initializes its own pipeline and processes sequences from the task queue.
    Results are sent back through the result queue with proper ordering information.
    
    Args:
        task_queue: Queue containing WorkerTask objects
        result_queue: Queue for returning ProcessingResult objects
        worker_id: Unique identifier for this worker process
    """
    pipeline = None
    processed_count = 0
    
    try:
        # Worker initialization
        process_name = current_process().name
        print(f"Worker {worker_id} ({process_name}): Starting up...")
        
        while True:
            try:
                # Get task from queue with longer timeout to allow for worker initialization
                task = task_queue.get(timeout=5.0)  # Increased from 1.0 to 5.0 seconds
                
                if task is None:  # Poison pill - shutdown signal
                    print(f"Worker {worker_id}: Received shutdown signal")
                    break
                
                # Initialize pipeline on first task (lazy initialization)
                if pipeline is None:
                    print(f"Worker {worker_id}: Initializing pipeline...")
                    pipeline = Vocab2EmbeddingPipeline(task.config_path)
                    pipeline.load_vocabulary(task.vocab_path)
                    pipeline.w_max = task.dynamic_w_max
                    
                    # Recreate candidate generator with updated w_max
                    span_config = pipeline.config.get('span_generation', {})
                    pipeline.candidate_generator = SpanCandidateGenerator(
                        pipeline.vocab_dict,
                        tau_vocab=float(span_config.get('tau_vocab', 1e-4)),
                        tau_comp=float(span_config.get('tau_comp', 1e-6)),
                        w_max=task.dynamic_w_max
                    )
                    
                    # GPU memory status after initialization
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"Worker {worker_id}: Pipeline initialized on {pipeline.device} "
                              f"[GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB res]")
                    else:
                        print(f"Worker {worker_id}: Pipeline initialized on {pipeline.device}")
                
                # Process the sequence
                start_time = time.time()
                try:
                    result = pipeline.process_sequence(task.sequence)
                    processing_time = time.time() - start_time
                    
                    # Add processing metadata
                    result['processing_time'] = processing_time
                    result['worker_id'] = worker_id
                    result['sequence'] = task.sequence  # Include original sequence for saving
                    
                    # CRITICAL: Detach tensors for multiprocessing serialization
                    # PyTorch tensors with requires_grad=True cannot be serialized across processes
                    # Move tensors to CPU to prevent GPU memory accumulation in shared memory
                    result_for_queue = {}
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            # Detach and move to CPU to avoid shared GPU memory issues
                            # Clone to completely separate from original computation graph
                            result_for_queue[key] = value.detach().cpu().clone()
                        else:
                            result_for_queue[key] = value
                    
                    # Clear original result to free GPU memory immediately
                    del result
                    
                    # Force immediate GPU memory cleanup after each sequence
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Send successful result with retry mechanism for Windows shared memory issues
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            result_queue.put(ProcessingResult(task.seq_id, True, result_for_queue))
                            break  # Success
                        except Exception as queue_error:
                            if attempt < max_retries - 1:
                                print(f"Worker {worker_id}: Queue put failed (attempt {attempt + 1}/{max_retries}), retrying: {queue_error}")
                                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                                continue
                            else:
                                # Final attempt failed - send error instead
                                error_msg = f"Worker {worker_id}: Failed to send result after {max_retries} attempts: {queue_error}"
                                try:
                                    result_queue.put(ProcessingResult(task.seq_id, False, error=error_msg))
                                except:
                                    print(f"Worker {worker_id}: Critical - Cannot send any result for sequence {task.seq_id}")
                                raise queue_error
                    
                    processed_count += 1
                    
                    # GPU memory monitoring for debugging
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                        peak = torch.cuda.max_memory_allocated() / 1024**3   # GB
                    else:
                        allocated = reserved = peak = 0.0
                    
                    print(f"Worker {worker_id}: Completed sequence {task.seq_id} "
                          f"({len(task.sequence)} chars, {processing_time:.2f}s, "
                          f"{result_for_queue['num_candidates']} candidates) "
                          f"[GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB res, {peak:.2f}GB peak]")
                    
                except Exception as e:
                    # Send error result with retry mechanism
                    error_msg = f"Worker {worker_id} error processing sequence {task.seq_id}: {str(e)}"
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            result_queue.put(ProcessingResult(task.seq_id, False, error=error_msg))
                            break  # Success
                        except Exception as queue_error:
                            if attempt < max_retries - 1:
                                print(f"Worker {worker_id}: Error queue put failed (attempt {attempt + 1}/{max_retries}), retrying: {queue_error}")
                                time.sleep(0.1 * (attempt + 1))
                                continue
                            else:
                                print(f"Worker {worker_id}: Critical - Cannot send error result for sequence {task.seq_id}: {queue_error}")
                                break
                    print(error_msg)
                
            except Empty:
                # Timeout waiting for task - check if we should continue
                continue
            except Exception as e:
                print(f"Worker {worker_id}: Unexpected error: {e}")
                break
                
    except KeyboardInterrupt:
        print(f"Worker {worker_id}: Received interrupt signal")
    except Exception as e:
        print(f"Worker {worker_id}: Fatal error: {e}")
    finally:
        # GPU memory status before cleanup
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            peak_before = torch.cuda.max_memory_allocated() / 1024**3
        else:
            allocated_before = reserved_before = peak_before = 0.0
        
        # Cleanup GPU memory and pipeline resources
        print(f"Worker {worker_id}: Cleaning up GPU memory and shutting down (processed {processed_count} sequences)")
        print(f"Worker {worker_id}: GPU before cleanup - {allocated_before:.2f}GB alloc, {reserved_before:.2f}GB res, {peak_before:.2f}GB peak")
        
        if pipeline is not None:
            # Clear all pipeline components to release GPU memory
            if hasattr(pipeline, 'unigram_lm') and pipeline.unigram_lm is not None:
                del pipeline.unigram_lm
            if hasattr(pipeline, 'seed_embedder') and pipeline.seed_embedder is not None:
                del pipeline.seed_embedder
            if hasattr(pipeline, 'conv_encoder') and pipeline.conv_encoder is not None:
                del pipeline.conv_encoder
            if hasattr(pipeline, 'candidate_generator') and pipeline.candidate_generator is not None:
                del pipeline.candidate_generator
            del pipeline
            pipeline = None
        
        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            print(f"Worker {worker_id}: GPU after cleanup - {allocated_after:.2f}GB alloc, {reserved_after:.2f}GB res")
        
        print(f"Worker {worker_id}: GPU cleanup completed")

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
        """
        Check if span has compositional segmentation potential following paper Equation:
        ∃ seg ∈ Segments(x,i,j) : ∏_{u ∈ seg} p(u) ≥ τ_comp
        
        This finds the OPTIMAL segmentation using dynamic programming (like Viterbi)
        and checks if its probability product meets the threshold.
        
        Optimized for performance while maintaining mathematical correctness.
        """
        # Fast early exits
        if not span_text or len(span_text) > 20:  # Skip very long spans
            return False
            
        if len(span_text) == 1:  # Single chars are trivially segmentable if in vocab
            if span_text in self._vocab_set:
                prob = self.vocab_dict[span_text]
                return isinstance(prob, (int, float)) and prob >= self.tau_comp
            return False
        
        # Quick vocabulary check first (fastest path - single piece)
        if span_text in self._vocab_set:
            prob = self.vocab_dict[span_text]
            if isinstance(prob, (int, float)) and prob >= self.tau_comp:
                return True
        
        # For multi-character spans, find optimal segmentation using dynamic programming
        # This is the correct implementation of the paper's mathematical specification
        span_len = len(span_text)
        
        # DP array: best_prob[i] = maximum probability to segment span_text[:i]
        best_prob = [0.0] * (span_len + 1)
        best_prob[0] = 1.0  # Empty prefix has probability 1
        
        for i in range(1, span_len + 1):
            # Try all possible previous positions j where we can place a piece
            for j in range(i):
                if best_prob[j] == 0.0:  # No valid segmentation to position j
                    continue
                    
                piece = span_text[j:i]
                if piece in self._vocab_set:
                    prob = self.vocab_dict[piece]
                    if isinstance(prob, (int, float)) and prob > 0:
                        # Update best probability: best_prob[j] * p(piece)
                        candidate_prob = best_prob[j] * prob
                        best_prob[i] = max(best_prob[i], candidate_prob)
        
        # Check if the optimal segmentation meets the threshold
        optimal_prob = best_prob[span_len]
        return optimal_prob >= self.tau_comp
    
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
        
        # Single-threaded processing with early termination optimizations
        for i in range(T):
            # Check for shutdown signal
            if SHUTDOWN_REQUESTED:
                if logger:
                    logger.warning("Shutdown requested during candidate generation")
                break
                
            max_j = min(i + self.w_max, T)
            
            for j in range(i + 1, max_j + 1):
                span_text = sequence[i:j]
                
                # Combined filtering with early termination (fastest to slowest)
                if (self.vocabulary_alignment(span_text) or
                    self.whitespace_coherent(span_text) or
                    self.compositional_potential(span_text)):
                    candidates.append((i, j))
        
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
        self.workers = processing_config.get('workers', 1)
        
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
        
        Following paper Equation: w_max = max(max_word_length, ⌊L_max/2⌋)
        where max_word_length is found by analyzing the corpus for the longest complete word.
        This ensures span generation is adapted to actual corpus content while respecting sequence limits.
        
        Args:
            sequences: List of input sequences from the corpus
            
        Returns:
            Computed w_max value (MAXIMUM of corpus-based and sequence-based bounds per paper)
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
        
        # Paper's specification: w_max = min(longest_word_length, ⌊L_max/2⌋)
        # This enforces a hard computational bound while allowing overlapping spans
        # to handle longer words through gated fusion (Section 3.6-3.8)
        corpus_based_w_max = max_word_length
        sequence_based_w_max = self.w_max_bound  # max_sequence_length // 2
        
        # Use the MINIMUM value to enforce computational bound - long words 
        # are handled by overlapping spans with gated fusion downstream
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
            'soft_probabilities': soft_probs,  # Keep on GPU until saving
            'seed_embeddings': seed_embeddings,  # Keep on GPU until saving  
            'contextual_embeddings': contextual_embeddings,  # Keep on GPU until saving
            'span_candidates': candidates,
            'sequence_length': len(sequence),
            'num_candidates': len(candidates),
            'span_width': span_width
        }
        
        # Add analysis if enabled
        if self.add_analysis:
            try:
                from x_spanformer.embedding.embedding_utils import analyze_embedding_quality
                # Try to keep on GPU first, fallback to CPU if needed
                try:
                    result['analysis'] = analyze_embedding_quality(contextual_embeddings)
                except (TypeError, RuntimeError):
                    # Function requires CPU tensors
                    result['analysis'] = analyze_embedding_quality(
                        contextual_embeddings.detach().cpu().numpy()
                    )
                get_embedding_logger('vocab2embedding').debug("Added embedding quality analysis to results")
            except Exception as e:
                get_embedding_logger('vocab2embedding').warning(f"Error during embedding analysis: {e}")
        
        return result


def process_sequences_parallel(sequences: List[str], missing_seq_ids: List[int], 
                             pipeline: 'Vocab2EmbeddingPipeline', num_workers: int,
                             chunk_manager: ChunkManager) -> Tuple[int, int]:
    """
    Process sequences in parallel while maintaining sequential output ordering and chunked storage.
    
    Args:
        sequences: List of all sequences from input corpus
        missing_seq_ids: List of sequence IDs that need processing
        pipeline: Configured pipeline instance (for config/vocab paths)
        num_workers: Number of worker processes
        chunk_manager: ChunkManager instance for chunked storage
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    if num_workers <= 1:
        return process_sequences_sequential(sequences, missing_seq_ids, pipeline, chunk_manager)
    
    logger = get_embedding_logger('vocab2embedding')
    
    logger.info(f"Starting parallel processing with {num_workers} workers")
    logger.info(f"Processing {len(missing_seq_ids)} sequences...")
    logger.info(f"Chunk size: {chunk_manager.chunk_size} sequences per chunk")
    
    # Create queues for task distribution and result collection
    # Use larger maxsize to prevent blocking on task population
    task_queue = Queue(maxsize=max(50, num_workers * 10))  # Larger queue to prevent blocking
    result_queue = Queue(maxsize=num_workers * 2)  # Keep result queue smaller
    
    # Start worker processes FIRST before populating queue
    workers = []
    for worker_id in range(num_workers):
        worker = Process(
            target=sequence_processor_worker,
            args=(task_queue, result_queue, worker_id),
            name=f"SeqWorker-{worker_id}"
        )
        worker.start()
        workers.append(worker)
    
    logger.info(f"Started {len(workers)} workers")
    
    # Give workers a moment to start up before flooding the queue
    import time
    import threading
    time.sleep(1.0)
    logger.info("Workers startup delay completed")
    
    # Use threading to handle queueing and result collection concurrently
    tasks_queued = 0
    failed_queue_attempts = 0
    queueing_complete = threading.Event()
    
    def producer_thread():
        """Thread to queue tasks while main thread collects results."""
        nonlocal tasks_queued, failed_queue_attempts
        
        max_failed_attempts = 10
        
        logger.info("Producer thread: Starting task queueing...")
        
        for seq_id in missing_seq_ids:
            if SHUTDOWN_REQUESTED or queueing_complete.is_set():
                break
                
            if failed_queue_attempts >= max_failed_attempts:
                logger.error(f"Too many failed queue attempts ({failed_queue_attempts}), stopping task queuing")
                break
                
            sequence = sequences[seq_id - 1]  # Convert to 0-based index
            task = WorkerTask(
                seq_id=seq_id,
                sequence=sequence,
                config_path=pipeline.config.get('_config_path', 'config/pipelines/vocab2embedding.yaml'),
                vocab_path=pipeline.config.get('_vocab_path', ''),
                dynamic_w_max=pipeline.w_max
            )
            
            # Adaptive queueing: wait for queue space before attempting
            task_queued = False
            max_wait_time = 30.0  # Maximum wait time for queue space
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait_time:
                if SHUTDOWN_REQUESTED or queueing_complete.is_set():
                    break
                
                try:
                    # Check queue size first
                    current_queue_size = task_queue.qsize()
                    queue_capacity = max(50, num_workers * 10)  # Use the same value as queue creation
                    queue_usage = current_queue_size / queue_capacity if queue_capacity > 0 else 0
                    
                    # Only attempt to queue if there's reasonable space (< 90% full)
                    if queue_usage < 0.9:
                        task_queue.put(task, timeout=0.1)  # Very short timeout
                        task_queued = True
                        tasks_queued += 1
                        if tasks_queued % 25 == 0:  # Progress updates
                            logger.info(f"Producer: Queued {tasks_queued}/{len(missing_seq_ids)} tasks (queue: {current_queue_size}/{queue_capacity})")
                        break  # Success, move to next task
                    else:
                        # Queue is too full, wait a bit before checking again
                        time.sleep(0.2)
                        
                except Full:
                    # Queue is full, wait before retrying
                    time.sleep(0.5)
                    continue
                    
                except Exception as e:
                    logger.error(f"Producer: Failed to queue task for sequence {seq_id}: {type(e).__name__}: {e}")
                    failed_queue_attempts += 1
                    break  # Move to next task
            
            if not task_queued:
                logger.warning(f"Producer: Could not queue task for sequence {seq_id} within {max_wait_time}s")
                failed_queue_attempts += 1
        
        # Add poison pills for workers (wait for queue space)
        logger.info("Producer: Queueing poison pills...")
        poison_pills_added = 0
        for i in range(num_workers):
            if SHUTDOWN_REQUESTED or queueing_complete.is_set():
                break
                
            # Wait for queue space before adding poison pill
            max_wait = 10.0
            wait_start = time.time()
            while time.time() - wait_start < max_wait:
                try:
                    task_queue.put(None, timeout=0.1)
                    poison_pills_added += 1
                    break
                except Full:
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    logger.warning(f"Producer: Failed to add poison pill {i+1}: {e}")
                    break
        
        # Report final queueing status
        if failed_queue_attempts > 0:
            logger.warning(f"Producer: Task queueing completed with {failed_queue_attempts} failed attempts")
        logger.info(f"Producer: Successfully queued {tasks_queued}/{len(missing_seq_ids)} tasks and {poison_pills_added}/{num_workers} poison pills")
        logger.info("Producer thread: Task queueing completed")
    
    # Start producer thread
    producer = threading.Thread(target=producer_thread, name="TaskProducer")
    producer.daemon = True  # Dies when main thread dies
    producer.start()
    logger.info("Started producer thread for task queueing")
    
    # Result collection with sequential ordering and chunked storage
    results_buffer = {}  # seq_id -> ProcessingResult
    processed_results = set()  # Track which sequence IDs have been saved
    processed_count = 0
    error_count = 0
    # We'll update expected_results when producer finishes
    expected_results = 0
    producer_finished = False
    
    logger.info("Collecting results and maintaining sequential order...")
    
    # Wait for producer to start and give initial status
    time.sleep(0.5)
    
    while not SHUTDOWN_REQUESTED:
        try:
            # Get result from queue with timeout
            result = result_queue.get(timeout=2.0)
            results_buffer[result.seq_id] = result
            logger.debug(f"Received result for sequence {result.seq_id} (success: {result.success})")
            
            # Process results in sequential order based on missing_seq_ids
            while True:
                # Find the next expected sequence ID to process
                next_seq_id = None
                for seq_id in missing_seq_ids:
                    if seq_id in results_buffer and seq_id not in processed_results:
                        next_seq_id = seq_id
                        break
                
                if next_seq_id is None:
                    break  # Wait for more results
                
                # Process the next sequential result
                seq_result = results_buffer[next_seq_id]
                processed_results.add(next_seq_id)
                
                if seq_result.success:
                    try:
                        # Log progress - calculate based on total sequences processed vs total sequences
                        total_processed = next_seq_id  # Current sequence ID represents total processed so far
                        progress = (total_processed / len(sequences)) * 100
                        logger.info(f"Sequence {next_seq_id}/{len(sequences)} completed "
                                  f"(Worker {seq_result.result.get('worker_id', '?')}, "
                                  f"{seq_result.result.get('processing_time', 0):.2f}s, "
                                  f"{progress:.1f}% complete)")
                        
                        processed_count += 1
                        
                        # Process each sequence individually with contiguous chunking
                        single_result = {next_seq_id: seq_result.result}
                        saved_chunks = save_sequence_individually_chunked(
                            chunk_manager, single_result, pipeline.config, logger
                        )
                        
                        # Periodic GPU memory cleanup every 10 sequences
                        if processed_count % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                    except Exception as e:
                        logger.error(f"Error processing results for sequence {next_seq_id}: {e}")
                        error_count += 1
                else:
                    logger.error(seq_result.error)
                    error_count += 1
        
        except Empty:
            # Check if producer thread is done and update expected_results
            if not producer.is_alive() and not producer_finished:
                producer_finished = True
                expected_results = tasks_queued
                logger.info(f"Producer finished: expecting {expected_results} results (processed: {processed_count}, errors: {error_count})")
            
            # If producer is finished and we've collected all expected results, we can exit
            if producer_finished and processed_count + error_count >= expected_results:
                logger.info(f"All expected results collected: {processed_count} processed, {error_count} errors")
                break
            
            # Check if workers are still alive and handle dead workers
            alive_workers = [w for w in workers if w.is_alive()]
            dead_workers = [w for w in workers if not w.is_alive()]
            
            if dead_workers:
                for dead_worker in dead_workers:
                    logger.warning(f"Worker {dead_worker.name} died unexpectedly (likely tensor serialization issue)")
                    # Check if it terminated with an error
                    if dead_worker.exitcode and dead_worker.exitcode != 0:
                        logger.error(f"Worker {dead_worker.name} exited with code {dead_worker.exitcode}")
            
            # If all workers are dead and no producer, and result queue is empty, exit
            if not alive_workers and producer_finished and result_queue.empty():
                logger.warning("All workers finished and result queue is empty")
                if expected_results > 0:
                    missing_results = expected_results - (processed_count + error_count)
                    if missing_results > 0:
                        logger.warning(f"Missing {missing_results} results - likely due to worker crashes")
                break
            
            # Continue waiting for more results
            continue
        except Exception as e:
            logger.error(f"Error collecting results: {e}")
            break
    
    # Wait for producer thread to finish (with timeout)
    if producer.is_alive():
        logger.info("Waiting for producer thread to finish...")
        queueing_complete.set()  # Signal producer to stop
        producer.join(timeout=5.0)
        if producer.is_alive():
            logger.warning("Producer thread did not finish cleanly")
    
    # Only flush remaining sequences if shutdown was NOT requested
    # This ensures we never save partial chunks during interruption
    if not SHUTDOWN_REQUESTED:
        final_chunks = chunk_manager.flush_remaining_sequences(pipeline.config)
        if final_chunks:
            for chunk_meta in final_chunks:
                logger.info(f"Final flush saved chunk {chunk_meta.chunk_id}: "
                           f"sequences {chunk_meta.start_seq_id}-{chunk_meta.end_seq_id}")
        else:
            logger.info("No remaining sequences to flush")
    else:
        # Count sequences in buffer that won't be saved
        buffer_count = len(chunk_manager.sequence_buffer)
        if buffer_count > 0:
            logger.info(f"Graceful shutdown: skipping {buffer_count} sequences in buffer "
                       f"(will be reprocessed on resume for complete chunks)")
        else:
            logger.info("Graceful shutdown: no partial sequences to discard")
    
    # Cleanup: wait for workers to finish with shorter timeout and force termination
    logger.info("Waiting for workers to finish...")
    for worker in workers:
        worker.join(timeout=5.0)  # Reduced timeout for faster shutdown
        if worker.is_alive():
            logger.warning(f"Worker {worker.name} did not finish cleanly - terminating forcefully")
            worker.terminate()
            worker.join(timeout=2.0)  # Brief wait for termination
            if worker.is_alive():
                logger.error(f"Worker {worker.name} failed to terminate - may require manual cleanup")
    
    # Additional cleanup for any remaining GPU resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info(f"Parallel processing completed: {processed_count} processed, {error_count} errors")
    return processed_count, error_count

def process_sequences_sequential(sequences: List[str], missing_seq_ids: List[int],
                               pipeline: 'Vocab2EmbeddingPipeline', chunk_manager: ChunkManager) -> Tuple[int, int]:
    """
    Process sequences sequentially using chunked storage (fallback for single worker or debugging).
    
    Args:
        sequences: List of all sequences from input corpus
        missing_seq_ids: List of sequence IDs that need processing  
        pipeline: Configured pipeline instance
        chunk_manager: ChunkManager instance for chunked storage
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    logger = get_embedding_logger('vocab2embedding')
    logger.info("Processing sequences sequentially...")
    logger.info(f"Chunk size: {chunk_manager.chunk_size} sequences per chunk")
    
    processed_count = 0
    error_count = 0
    
    for seq_id in missing_seq_ids:
        if SHUTDOWN_REQUESTED:
            logger.warning("SHUTDOWN SIGNAL RECEIVED - Stopping processing")
            break
            
        sequence = sequences[seq_id - 1]  # Convert to 0-based index
        
        try:
            logger.info(f"Sequence {seq_id}/{len(sequences)} - Length: {len(sequence)} chars")
            
            # GPU memory status for monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Process sequence using the pipeline
            result = pipeline.process_sequence(sequence)
            
            logger.info(f"  Forward-backward: {result['soft_probabilities'].shape}")
            logger.info(f"  Seed embeddings: {result['seed_embeddings'].shape}")
            logger.info(f"  Contextual embeddings: {result['contextual_embeddings'].shape}")
            logger.info(f"  Span candidates: {result['num_candidates']} (span_width: {result['span_width']})")
            
            # Add sequence text for storage
            result['sequence'] = sequence
            
            processed_count += 1
            
            # Process sequence individually with contiguous chunking
            single_result = {seq_id: result}
            saved_chunks = save_sequence_individually_chunked(
                chunk_manager, single_result, pipeline.config, logger
            )
            
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Sequence {seq_id}: Error processing sequence: {e}")
            error_count += 1
            continue
    
    # Only flush remaining sequences if shutdown was NOT requested
    # This ensures we never save partial chunks during interruption  
    if not SHUTDOWN_REQUESTED:
        final_chunks = chunk_manager.flush_remaining_sequences(pipeline.config)
        if final_chunks:
            for chunk_meta in final_chunks:
                logger.info(f"Final sequential flush saved chunk {chunk_meta.chunk_id}: "
                           f"sequences {chunk_meta.start_seq_id}-{chunk_meta.end_seq_id}")
        else:
            logger.info("No remaining sequences to flush")
    else:
        # Count sequences in buffer that won't be saved
        buffer_count = len(chunk_manager.sequence_buffer)
        if buffer_count > 0:
            logger.info(f"Graceful shutdown: skipping {buffer_count} sequences in buffer "
                       f"(will be reprocessed on resume for complete chunks)")
        else:
            logger.info("Graceful shutdown: no partial sequences to discard")
    
    return processed_count, error_count


def log_saved_files(file_sizes, logger):
    """Log saved file information in a clean, readable format."""
    # Build components list
    components = []
    total_size = 0
    
    # Add components in consistent order
    for component, key in [('JSON', 'json'), ('Seed', 'seed'), ('Context', 'context'), ('SoftProb', 'soft_prob')]:
        if key in file_sizes:
            size_kb = file_sizes[key]
            components.append(f"{component}({size_kb:.1f}KB)")
            total_size += size_kb
        else:
            # Skip rather than show "SKIPPED" for cleaner output
            pass
    
    # Log the result
    if components:
        components_str = " + ".join(components)
        logger.info(f"  Saved: {components_str} = {total_size:.1f}KB total")
    else:
        logger.info(f"  Saved: Context({file_sizes.get('context', 0):.1f}KB) = {total_size:.1f}KB total")


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
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes for parallel sequence processing (default: 1)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of sequences per chunk file (overrides config, default: 100)"
    )
    
    return parser.parse_args()


def find_missing_sequences(total_sequences: int, existing_records: Dict[int, Dict]) -> List[int]:
    """
    Find missing sequence IDs that need to be processed.
    
    This handles discontinuous completions where sequences might be completed
    out of order (e.g., 1-10 done, 12 done, but 11 and 13+ need processing).
    
    Args:
        total_sequences: Total number of sequences in the input corpus
        existing_records: Dictionary of already processed sequence records
        
    Returns:
        List of sequence IDs that need to be processed (sorted)
    """
    all_sequence_ids = set(range(1, total_sequences + 1))
    completed_sequence_ids = set(existing_records.keys())
    missing_sequence_ids = sorted(all_sequence_ids - completed_sequence_ids)
    
    logger = get_embedding_logger('vocab2embedding')
    if missing_sequence_ids:
        logger.info(f"Missing sequences detected: {len(missing_sequence_ids)} sequences need processing")
        
        # Show ranges for cleaner logging
        ranges = []
        start = missing_sequence_ids[0]
        end = start
        
        for seq_id in missing_sequence_ids[1:]:
            if seq_id == end + 1:
                end = seq_id
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = seq_id
        
        # Add final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        logger.info(f"Missing sequence ranges: {', '.join(ranges)}")
    else:
        logger.info("All sequences already processed - no missing sequences detected")
    
    return missing_sequence_ids

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
    logger.info(f"  Workers: {args.workers}")
    logger.info("-" * 80)
    
    logger.info("[bold cyan]X-Spanformer VOCAB2EMBEDDING Pipeline[/bold cyan]")
    logger.info("[green]Initializing embedding generation pipeline[/green]")
    
    # Initialize pipeline and log full configuration FIRST
    logger.info("=" * 50)
    logger.info("STAGE 1: PIPELINE INITIALIZATION")
    logger.info("=" * 50)
    
    pipeline = Vocab2EmbeddingPipeline(args.config)
    
    # Store config and vocab paths for worker processes
    pipeline.config['_config_path'] = args.config
    pipeline.config['_vocab_path'] = args.vocab
    
    # Override config with CLI arguments if provided
    if args.workers != 1:  # Only override if different from default
        pipeline.workers = args.workers
        logger.info(f"Workers setting overridden by CLI: {args.workers}")
    
    # Handle chunk size override
    chunk_size = args.chunk_size if args.chunk_size is not None else pipeline.config.get('output', {}).get('chunk_size', 100)
    if args.chunk_size is not None:
        logger.info(f"Chunk size overridden by CLI: {chunk_size}")
        pipeline.config['output']['chunk_size'] = chunk_size
    
    # Initialize chunk manager
    chunk_manager = ChunkManager(output_path, chunk_size)
    
    # Load initial existing records for resume detection
    existing_sequences = chunk_manager.get_existing_sequences()
    existing_records = {seq_id: {'sequence_id': seq_id} for seq_id in existing_sequences}
    last_processed = max(existing_sequences) if existing_sequences else 0
    if existing_records:
        logger.info(f"RESUMING PROCESSING - Found {len(existing_records)} previously processed sequences")
        logger.info(f"Last processed sequence ID: {last_processed}")
        logger.info(f"Existing chunks: {len(chunk_manager.chunks_metadata)}")
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
    
    logger.info(f"Workers: {pipeline.workers}")
    logger.info(f"Pipeline initialized on: {pipeline.device}")
    logger.info("-" * 50)

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
    
    # Validate existing chunks and repair if needed BEFORE determining missing sequences
    if existing_records:
        logger.info("=" * 50)
        logger.info("STAGE 4.5: CHUNK VALIDATION AND REPAIR")
        logger.info("=" * 50)
        
        logger.info(f"Validating {len(chunk_manager.chunks_metadata)} existing chunks...")
        
        # Get all missing sequences and chunk gaps
        all_missing, chunk_gaps = chunk_manager.validate_and_get_missing_sequences(len(sequences))
        
        if chunk_gaps:
            logger.warning(f"Found {len(chunk_gaps)} incomplete chunks that need repair")
            
            # Add option to skip repair if it causes issues
            skip_repair = False
            try:
                # Test pipeline processing first with a simple sequence
                logger.info("Testing pipeline before chunk repair...")
                test_sequence = "test"
                test_result = pipeline.process_sequence(test_sequence)
                logger.info("Pipeline test successful - proceeding with chunk repair")
                
                # Clear test result from GPU
                if torch.cuda.is_available():
                    del test_result
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Pipeline test failed: {e}")
                logger.warning("Skipping chunk repair due to pipeline issues - will process all missing sequences instead")
                skip_repair = True
            
            if not skip_repair:
                # Create a sequence processor function for chunk repair
                def repair_sequence_processor(seq_id: int):
                    """Process a single sequence for chunk repair."""
                    repair_logger = get_embedding_logger('vocab2embedding')
                    try:
                        repair_logger.info(f"Repair: Starting sequence {seq_id}")
                        
                        if seq_id <= len(sequences):
                            sequence = sequences[seq_id - 1]  # Convert to 0-based index
                            repair_logger.info(f"Repair: Processing sequence {seq_id} ({len(sequence)} chars)")
                            
                            # Check pipeline components before processing
                            if pipeline.unigram_lm is None or pipeline.seed_embedder is None or pipeline.conv_encoder is None:
                                repair_logger.error(f"Repair: Pipeline components not properly initialized")
                                return None
                            
                            # Check CUDA availability
                            if pipeline.device.startswith('cuda') and not torch.cuda.is_available():
                                repair_logger.error(f"Repair: CUDA device specified but not available")
                                return None
                            
                            repair_logger.info(f"Repair: Calling pipeline.process_sequence for {seq_id}")
                            result = pipeline.process_sequence(sequence)
                            repair_logger.info(f"Repair: Pipeline processing completed for {seq_id}")
                            
                            # Convert GPU tensors to CPU numpy arrays for storage
                            processed_result = {
                                'sequence': sequence,
                                'span_candidates': result['span_candidates'],
                                'contextual_embeddings': result['contextual_embeddings'].detach().cpu().numpy()
                            }
                            
                            # Conditionally add other components based on config
                            output_config = pipeline.config.get('output', {})
                            if output_config.get('save_seed_embeddings', False):
                                processed_result['seed_embeddings'] = result['seed_embeddings'].detach().cpu().numpy()
                            if output_config.get('save_soft_probabilities', False):
                                processed_result['soft_probabilities'] = result['soft_probabilities'].detach().cpu().numpy()
                            
                            repair_logger.info(f"Repair: processed sequence {seq_id} ({len(sequence)} chars, {result['num_candidates']} candidates)")
                            return processed_result
                        else:
                            repair_logger.error(f"Repair: sequence {seq_id} out of range (max: {len(sequences)})")
                            return None
                    except Exception as e:
                        repair_logger.error(f"Repair: failed to process sequence {seq_id}: {e}")
                        import traceback
                        repair_logger.error(f"Repair: traceback: {traceback.format_exc()}")
                        return None
                
                # Repair incomplete chunks with timeout protection
                logger.info(f"Repairing {len(chunk_gaps)} incomplete chunks...")
                repair_start_time = time.time()
                
                try:
                    repair_success = chunk_manager.repair_incomplete_chunks(
                        chunk_gaps, repair_sequence_processor, pipeline.config
                    )
                    
                    repair_duration = time.time() - repair_start_time
                    logger.info(f"Chunk repair completed in {repair_duration:.2f}s")
                    
                    if not repair_success:
                        logger.error("Some chunk repairs failed - proceeding but data integrity may be compromised")
                    else:
                        logger.info("All chunk repairs completed successfully")
                        
                    # Reload existing records after repairs
                    existing_sequences = chunk_manager.get_existing_sequences()
                    existing_records = {seq_id: {'sequence_id': seq_id} for seq_id in existing_sequences}
                    last_processed = max(existing_sequences) if existing_sequences else 0
                    logger.info(f"After repairs: {len(existing_records)} sequences processed, last ID: {last_processed}")
                    
                except Exception as e:
                    repair_duration = time.time() - repair_start_time
                    logger.error(f"Chunk repair failed after {repair_duration:.2f}s: {e}")
                    import traceback
                    logger.error(f"Repair traceback: {traceback.format_exc()}")
                    logger.warning("Proceeding without repairs - some chunks may remain incomplete")
            else:
                logger.info("Skipped chunk repair - will process all missing sequences instead")
        else:
            logger.info("All existing chunks are complete - no repairs needed")
    
    # Determine missing sequences for processing (handles discontinuous completions)
    missing_seq_ids = find_missing_sequences(len(sequences), existing_records)
    
    if not missing_seq_ids:
        logger.info("All sequences already processed - nothing to do!")
        logger.info(f"Output saved to: {output_path}")
        return
    
    # Process sequences from loaded data
    logger.info("=" * 50)
    logger.info("STAGE 5: SEQUENCE PROCESSING")
    logger.info("=" * 50)
    
    skipped_count = len(existing_records)  # Count previously processed as skipped
    
    logger.info(f"Processing {len(missing_seq_ids)} remaining sequences out of {len(sequences)} total...")
    logger.info(f"Using {pipeline.workers} worker{'s' if pipeline.workers > 1 else ''}...")
    
    # Process sequences (parallel or sequential based on worker count)
    processed_count, error_count = process_sequences_parallel(
        sequences, missing_seq_ids, pipeline, pipeline.workers, chunk_manager
    )
    
    # Perform final integrity check after processing
    logger.info("=" * 50)
    logger.info("STAGE 6: FINAL INTEGRITY VERIFICATION")
    logger.info("=" * 50)
    
    integrity_check_passed = chunk_manager.final_integrity_check(len(sequences))
    
    if not integrity_check_passed:
        logger.error("FINAL INTEGRITY CHECK FAILED - Some sequences may be missing!")
        logger.warning("Consider running the pipeline again to repair any missing sequences")
    else:
        logger.info("FINAL INTEGRITY CHECK PASSED - All sequences are present and accounted for")
    
    # Final statistics
    total_processed = skipped_count + processed_count
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 50)
    
    if SHUTDOWN_REQUESTED:
        logger.warning("Pipeline terminated by user request (graceful shutdown)")
    elif not integrity_check_passed:
        logger.warning("Pipeline completed with integrity issues - some sequences may be missing")
    else:
        logger.info("Pipeline completed successfully with all sequences verified")
    
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
    
    # Get chunk statistics
    chunks_dir = chunk_manager.chunks_dir
    chunk_files = list(chunks_dir.glob("embeddings_*.npz"))
    total_sequences_in_chunks = len(chunk_manager.get_existing_sequences())
    
    logger.info(f"  Chunks directory: {chunks_dir}")
    logger.info(f"  Chunk files: {len(chunk_files)} files")
    logger.info(f"  Total sequences in chunks: {total_sequences_in_chunks}")
    logger.info(f"  Chunk size: {chunk_manager.chunk_size} sequences per chunk")
    if chunk_files:
        logger.info(f"  Chunk file pattern: {chunk_files[0].name} ... {chunk_files[-1].name}")
    logger.info(f"  Metadata file: {chunk_manager.metadata_file}")
    logger.info(f"  Log file: {output_path / 'embedding.log'}")
    
    # Exit code for automation with forced termination to prevent hanging
    if SHUTDOWN_REQUESTED:
        logger.info("Exiting with code 130 (interrupted)")
        # Force cleanup and exit immediately to prevent hanging
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Use os._exit to bypass any cleanup that might hang
        import os
        os._exit(130)
    elif not integrity_check_passed:
        logger.info("Exiting with code 2 (integrity check failed)")
        sys.exit(2)
    elif error_count > 0:
        logger.info("Exiting with code 1 (errors encountered)")
        sys.exit(1)
    else:
        logger.info("Exiting with code 0 (success)")
        sys.exit(0)


if __name__ == "__main__":
    main()
