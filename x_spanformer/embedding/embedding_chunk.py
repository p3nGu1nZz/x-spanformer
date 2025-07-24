#!/usr/bin/env python3
"""
embedding_chunk.py

Chunked storage management for embedding data in the X-Spanformer pipeline.

This module provides efficient storage and retrieval of embedding data using
compressed chunks instead of individual files. Key features:

- Compressed .npz storage for space efficiency
- Configurable chunk sizes for optimal I/O performance  
- Metadata tracking with JSON registry
- Resume-friendly sequence detection
- Integrity verification and error recovery

The chunked storage system replaces thousands of individual .npy files with
a smaller number of compressed chunks, dramatically improving filesystem
performance and storage efficiency.

Usage:
    from x_spanformer.embedding.embedding_chunk import ChunkManager
    
    # Initialize chunk manager
    chunk_manager = ChunkManager(output_dir, chunk_size=100)
    
    # Save sequence results in batches
    chunk_meta = chunk_manager.save_chunk(result_buffer, pipeline_config)
    
    # Load existing sequences for resume
    existing_sequences = chunk_manager.get_existing_sequences()
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from x_spanformer.embedding.embedding_logging import get_embedding_logger


@dataclass
class ChunkMetadata:
    """
    Metadata for a single embedding chunk.
    
    Tracks essential information about each chunk including sequence ranges,
    file paths, creation timestamps, and component availability.
    """
    chunk_id: int
    start_seq_id: int  
    end_seq_id: int
    sequence_count: int
    file_path: str
    created_at: str
    components: List[str]  # ['context', 'seed', 'soft_prob', etc.]
    file_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary loaded from JSON."""
        return cls(**data)
    
    def contains_sequence(self, seq_id: int) -> bool:
        """Check if this chunk contains a specific sequence ID."""
        return self.start_seq_id <= seq_id <= self.end_seq_id
    
    def get_sequence_range(self) -> range:
        """Get range object for all sequence IDs in this chunk."""
        return range(self.start_seq_id, self.end_seq_id + 1)


class ChunkManager:
    """
    Manages chunked storage of embedding data.
    
    Provides high-level interface for saving and loading embedding data
    in compressed chunks. Handles metadata tracking, integrity verification,
    and resume functionality.
    
    Key Benefits:
    - Reduced filesystem overhead (fewer files)
    - Improved I/O performance through batching
    - Space savings via compression
    - Fast resume detection via metadata scanning
    """
    
    def __init__(self, output_dir: Path, chunk_size: int = 100):
        """
        Initialize chunk manager.
        
        Args:
            output_dir: Base output directory for embedding files
            chunk_size: Number of sequences per chunk (default: 100)
        """
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.json"
        self.chunks_metadata: Dict[int, ChunkMetadata] = {}
        self.logger = get_embedding_logger('embedding_chunk')
        
        # Buffer for accumulating sequences before saving complete chunks
        self.sequence_buffer: Dict[int, Dict] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        self.logger.debug(f"ChunkManager initialized: {len(self.chunks_metadata)} existing chunks")
    
    def _load_metadata(self) -> None:
        """Load existing chunk metadata from disk."""
        if not self.metadata_file.exists():
            self.logger.debug("No existing metadata file found")
            return
            
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                
            # Load chunk metadata
            chunk_data = data.get('chunks', {})
            self.chunks_metadata = {
                int(chunk_id): ChunkMetadata.from_dict(chunk_meta)
                for chunk_id, chunk_meta in chunk_data.items()
            }
            
            # Validate stored chunk_size matches current setting
            stored_chunk_size = data.get('chunk_size')
            if stored_chunk_size and stored_chunk_size != self.chunk_size:
                self.logger.warning(
                    f"Chunk size mismatch: stored={stored_chunk_size}, current={self.chunk_size}. "
                    "This may cause issues with resume functionality."
                )
            
            self.logger.info(f"Loaded metadata for {len(self.chunks_metadata)} chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to load chunk metadata: {e}")
            self.chunks_metadata = {}
    
    def _save_metadata(self) -> None:
        """Save chunk metadata to disk."""
        metadata = {
            'chunk_size': self.chunk_size,
            'total_chunks': len(self.chunks_metadata),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'chunks': {
                str(chunk_id): chunk_meta.to_dict()
                for chunk_id, chunk_meta in self.chunks_metadata.items()
            }
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.debug(f"Saved metadata for {len(self.chunks_metadata)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to save chunk metadata: {e}")
    
    def get_chunk_id(self, seq_id: int) -> int:
        """
        Get chunk ID for a given sequence ID.
        
        Uses deterministic mapping: chunk_id = ceil(seq_id / chunk_size)
        
        Args:
            seq_id: Sequence ID (1-based)
            
        Returns:
            Chunk ID (1-based)
        """
        return ((seq_id - 1) // self.chunk_size) + 1
    
    def get_chunk_range(self, chunk_id: int) -> Tuple[int, int]:
        """
        Get start and end sequence IDs for a chunk.
        
        Args:
            chunk_id: Chunk ID (1-based)
            
        Returns:
            Tuple of (start_seq_id, end_seq_id) both inclusive
        """
        start_seq_id = ((chunk_id - 1) * self.chunk_size) + 1
        end_seq_id = chunk_id * self.chunk_size
        return start_seq_id, end_seq_id
    
    def get_chunk_file_path(self, chunk_id: int) -> Path:
        """
        Get file path for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Path to chunk file (embeddings_XXXXXX.npz)
        """
        return self.chunks_dir / f"embeddings_{chunk_id:06d}.npz"
    
    def save_chunk(self, chunk_data: Dict[int, Dict], pipeline_config: Dict) -> Optional[ChunkMetadata]:
        """
        Save a chunk of embedding data to disk.
        
        If sequences span multiple chunks, they will be automatically grouped
        and saved to their appropriate chunks.
        
        Args:
            chunk_data: Dictionary mapping sequence_id -> result_dict
            pipeline_config: Pipeline configuration for determining what to save
            
        Returns:
            ChunkMetadata of the last chunk saved, None if no data saved
        """
        if not chunk_data:
            self.logger.debug("Empty chunk data, skipping save")
            return None
        
        # Group sequences by their target chunks
        seq_ids = sorted(chunk_data.keys())
        chunks_to_save = {}
        
        for seq_id in seq_ids:
            chunk_id = self.get_chunk_id(seq_id)
            if chunk_id not in chunks_to_save:
                chunks_to_save[chunk_id] = {}
            chunks_to_save[chunk_id][seq_id] = chunk_data[seq_id]
        
        # If sequences span multiple chunks, save each chunk separately
        if len(chunks_to_save) > 1:
            self.logger.debug(
                f"Sequences {seq_ids[0]}-{seq_ids[-1]} span {len(chunks_to_save)} chunks, "
                f"saving separately"
            )
            
            last_chunk_meta = None
            for chunk_id in sorted(chunks_to_save.keys()):
                chunk_meta = self._save_single_chunk(chunks_to_save[chunk_id], pipeline_config, chunk_id)
                if chunk_meta:
                    last_chunk_meta = chunk_meta
            return last_chunk_meta
        
        # Single chunk case
        chunk_id = list(chunks_to_save.keys())[0]
        return self._save_single_chunk(chunk_data, pipeline_config, chunk_id)
    
    def _save_single_chunk(self, chunk_data: Dict[int, Dict], pipeline_config: Dict, 
                          chunk_id: int) -> Optional[ChunkMetadata]:
        """
        Save a single chunk of data to disk.
        
        Args:
            chunk_data: Dictionary mapping sequence_id -> result_dict (all same chunk)
            pipeline_config: Pipeline configuration for determining what to save
            chunk_id: Target chunk ID
            
        Returns:
            ChunkMetadata if successful, None otherwise
        """
        seq_ids = sorted(chunk_data.keys())
        chunk_file = self.get_chunk_file_path(chunk_id)
        
        try:
            # Prepare data for saving
            arrays_to_save = {}
            components = []
            
            # Always save contextual embeddings (required)
            context_embeddings = []
            sequences = []
            span_candidates = []
            
            for seq_id in seq_ids:
                result = chunk_data[seq_id]
                context_embeddings.append(result['contextual_embeddings'])
                sequences.append(result['sequence'])
                span_candidates.append(result['span_candidates'])
            
            arrays_to_save['contextual_embeddings'] = np.array(context_embeddings, dtype=object)
            arrays_to_save['sequences'] = np.array(sequences, dtype=object)
            arrays_to_save['span_candidates'] = np.array(span_candidates, dtype=object)
            arrays_to_save['sequence_ids'] = np.array(seq_ids)
            components.append('context')
            
            # Conditionally save other components based on config
            output_config = pipeline_config.get('output', {})
            
            if output_config.get('save_seed_embeddings', False):
                seed_embeddings = []
                for seq_id in seq_ids:
                    if 'seed_embeddings' in chunk_data[seq_id]:
                        seed_embeddings.append(chunk_data[seq_id]['seed_embeddings'])
                    else:
                        self.logger.warning(f"Sequence {seq_id} missing seed_embeddings but config requires it")
                        return None
                arrays_to_save['seed_embeddings'] = np.array(seed_embeddings, dtype=object)
                components.append('seed')
            
            if output_config.get('save_soft_probabilities', False):
                soft_probs = []
                for seq_id in seq_ids:
                    if 'soft_probabilities' in chunk_data[seq_id]:
                        soft_probs.append(chunk_data[seq_id]['soft_probabilities'])
                    else:
                        self.logger.warning(f"Sequence {seq_id} missing soft_probabilities but config requires it")
                        return None
                arrays_to_save['soft_probabilities'] = np.array(soft_probs, dtype=object)
                components.append('soft_prob')
            
            # Save chunk file with compression
            np.savez_compressed(chunk_file, **arrays_to_save)
            
            # Calculate file size
            file_size_mb = chunk_file.stat().st_size / (1024 * 1024)
            
            # Create metadata
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                start_seq_id=seq_ids[0],
                end_seq_id=seq_ids[-1],
                sequence_count=len(seq_ids),
                file_path=str(chunk_file.relative_to(self.output_dir)),
                created_at=datetime.now().isoformat(),
                components=components,
                file_size_mb=round(file_size_mb, 2)
            )
            
            # Update metadata registry
            self.chunks_metadata[chunk_id] = chunk_meta
            self._save_metadata()
            
            self.logger.info(
                f"Saved chunk {chunk_id}: sequences {seq_ids[0]}-{seq_ids[-1]} "
                f"({len(seq_ids)} sequences, {file_size_mb:.1f}MB, components: {components})"
            )
            
            return chunk_meta
            
        except Exception as e:
            self.logger.error(f"Failed to save chunk {chunk_id}: {e}")
            return None
    
    def load_chunk(self, chunk_id: int) -> Optional[Dict[int, Dict]]:
        """
        Load a chunk of embedding data from disk.
        
        Args:
            chunk_id: Chunk ID to load
            
        Returns:
            Dictionary mapping sequence_id -> result_dict, or None if failed
        """
        if chunk_id not in self.chunks_metadata:
            self.logger.warning(f"Chunk {chunk_id} not found in metadata")
            return None
        
        chunk_file = self.get_chunk_file_path(chunk_id)
        if not chunk_file.exists():
            self.logger.error(f"Chunk file missing: {chunk_file}")
            return None
        
        try:
            data = np.load(chunk_file, allow_pickle=True)
            seq_ids = data['sequence_ids']
            
            chunk_data = {}
            for i, seq_id in enumerate(seq_ids):
                result = {
                    'sequence_id': int(seq_id),
                    'sequence': str(data['sequences'][i]),
                    'contextual_embeddings': data['contextual_embeddings'][i],
                    'span_candidates': [
                        tuple(span) if isinstance(span, list) else span
                        for span in (
                            data['span_candidates'][i].tolist() 
                            if hasattr(data['span_candidates'][i], 'tolist')
                            else data['span_candidates'][i]
                        )
                    ]
                }
                
                # Add optional components if they exist
                if 'seed_embeddings' in data:
                    result['seed_embeddings'] = data['seed_embeddings'][i]
                if 'soft_probabilities' in data:
                    result['soft_probabilities'] = data['soft_probabilities'][i]
                
                chunk_data[int(seq_id)] = result
            
            self.logger.debug(f"Loaded chunk {chunk_id}: {len(chunk_data)} sequences")
            return chunk_data
            
        except Exception as e:
            self.logger.error(f"Failed to load chunk {chunk_id}: {e}")
            return None
    
    def get_existing_sequences(self) -> Set[int]:
        """
        Get set of all existing sequence IDs across all chunks.
        
        Returns:
            Set of sequence IDs that have been processed and stored
        """
        existing_sequences = set()
        
        for chunk_meta in self.chunks_metadata.values():
            # Add all sequences in the range for this chunk
            existing_sequences.update(chunk_meta.get_sequence_range())
        
        self.logger.debug(f"Found {len(existing_sequences)} existing sequences across {len(self.chunks_metadata)} chunks")
        return existing_sequences
    
    def verify_chunk_integrity(self, chunk_id: int) -> bool:
        """
        Verify that a chunk file exists and is valid.
        
        Args:
            chunk_id: Chunk ID to verify
            
        Returns:
            True if chunk is valid, False otherwise
        """
        if chunk_id not in self.chunks_metadata:
            return False
        
        chunk_file = self.get_chunk_file_path(chunk_id)
        if not chunk_file.exists():
            return False
        
        try:
            # Quick validation - check if we can load the basic structure
            data = np.load(chunk_file, allow_pickle=True)
            required_keys = ['sequence_ids', 'contextual_embeddings', 'sequences', 'span_candidates']
            
            if not all(key in data for key in required_keys):
                return False
            
            # Verify sequence count matches metadata
            actual_count = len(data['sequence_ids'])
            expected_count = self.chunks_metadata[chunk_id].sequence_count
            
            if actual_count != expected_count:
                self.logger.warning(
                    f"Chunk {chunk_id} sequence count mismatch: "
                    f"actual={actual_count}, expected={expected_count}"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Chunk {chunk_id} integrity check failed: {e}")
            return False
    
    def verify_all_chunks(self) -> Dict[int, bool]:
        """
        Verify integrity of all chunks.
        
        Returns:
            Dictionary mapping chunk_id -> is_valid
        """
        results = {}
        for chunk_id in self.chunks_metadata.keys():
            results[chunk_id] = self.verify_chunk_integrity(chunk_id)
        
        valid_count = sum(results.values())
        total_count = len(results)
        
        if valid_count == total_count:
            self.logger.info(f"All {total_count} chunks are valid")
        else:
            invalid_count = total_count - valid_count
            self.logger.warning(f"{invalid_count}/{total_count} chunks failed integrity check")
        
        return results
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about stored chunks.
        
        Returns:
            Dictionary with chunk statistics
        """
        if not self.chunks_metadata:
            return {
                'total_chunks': 0,
                'total_sequences': 0,
                'total_size_mb': 0.0,
                'avg_chunk_size': 0.0,
                'chunk_range': (0, 0)
            }
        
        total_chunks = len(self.chunks_metadata)
        total_sequences = sum(meta.sequence_count for meta in self.chunks_metadata.values())
        total_size_mb = sum(meta.file_size_mb for meta in self.chunks_metadata.values())
        avg_chunk_size = total_sequences / total_chunks if total_chunks > 0 else 0
        
        chunk_ids = list(self.chunks_metadata.keys())
        chunk_range = (min(chunk_ids), max(chunk_ids))
        
        return {
            'total_chunks': total_chunks,
            'total_sequences': total_sequences,
            'total_size_mb': round(total_size_mb, 2),
            'avg_chunk_size': round(avg_chunk_size, 1),
            'chunk_range': chunk_range,
            'chunk_size_setting': self.chunk_size
        }
    
    def cleanup_orphaned_files(self) -> List[Path]:
        """
        Remove chunk files that are not tracked in metadata.
        
        Returns:
            List of files that were removed
        """
        if not self.chunks_dir.exists():
            return []
        
        # Get all chunk files
        chunk_files = list(self.chunks_dir.glob("embeddings_*.npz"))
        tracked_files = {self.get_chunk_file_path(chunk_id) for chunk_id in self.chunks_metadata.keys()}
        
        orphaned_files = [f for f in chunk_files if f not in tracked_files]
        
        for orphaned_file in orphaned_files:
            try:
                orphaned_file.unlink()
                self.logger.info(f"Removed orphaned chunk file: {orphaned_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to remove orphaned file {orphaned_file}: {e}")
        
        return orphaned_files
    
    def add_sequence_to_buffer(self, seq_id: int, sequence_data: Dict, pipeline_config: Optional[Dict] = None) -> List[ChunkMetadata]:
        """
        Add a sequence to the buffer and save any complete chunks.
        
        This method accumulates sequences and only saves complete, contiguous chunks
        of the specified chunk_size. This ensures consistent chunk sizes and faster
        loading performance.
        
        Args:
            seq_id: Sequence ID
            sequence_data: Sequence result data
            pipeline_config: Pipeline configuration (uses minimal config if None)
            
        Returns:
            List of ChunkMetadata objects for any chunks that were saved
        """
        if pipeline_config is None:
            pipeline_config = {'output': {}}
            
        # Add sequence to buffer
        self.sequence_buffer[seq_id] = sequence_data
        
        saved_chunks = []
        
        # Check if we can save any complete chunks
        while True:
            # Find the lowest chunk that could be complete
            buffered_ids = set(self.sequence_buffer.keys())
            if not buffered_ids:
                break
                
            # Get the lowest chunk ID that has sequences in buffer
            min_seq_id = min(buffered_ids)
            chunk_id = self.get_chunk_id(min_seq_id)
            
            # Get the expected range for this chunk
            start_seq_id, end_seq_id = self.get_chunk_range(chunk_id)
            expected_sequences = set(range(start_seq_id, end_seq_id + 1))
            
            # Check if we have all sequences for this chunk
            buffered_for_chunk = buffered_ids & expected_sequences
            
            if len(buffered_for_chunk) == self.chunk_size:
                # We have a complete chunk, save it
                chunk_data = {
                    seq_id: self.sequence_buffer[seq_id] 
                    for seq_id in expected_sequences 
                    if seq_id in self.sequence_buffer
                }
                
                # Save the chunk using the single chunk method directly
                chunk_meta = self._save_single_chunk(chunk_data, pipeline_config, chunk_id)
                if chunk_meta:
                    saved_chunks.append(chunk_meta)
                
                # Remove saved sequences from buffer
                for seq_id in expected_sequences:
                    self.sequence_buffer.pop(seq_id, None)
                    
                self.logger.info(
                    f"Saved complete chunk {chunk_id}: sequences {start_seq_id}-{end_seq_id} "
                    f"({self.chunk_size} sequences)"
                )
            else:
                # Cannot complete this chunk yet
                break
        
        return saved_chunks
    
    def flush_remaining_sequences(self, pipeline_config: Dict) -> List[ChunkMetadata]:
        """
        Save any remaining sequences in buffer as partial chunks.
        
        This should be called at the end of processing to save any sequences
        that don't form complete chunks.
        
        Args:
            pipeline_config: Pipeline configuration for determining what to save
            
        Returns:
            List of ChunkMetadata objects for any chunks that were saved
        """
        if not self.sequence_buffer:
            return []
        
        saved_chunks = []
        
        # Group remaining sequences by chunk
        chunks_to_save = {}
        for seq_id in self.sequence_buffer.keys():
            chunk_id = self.get_chunk_id(seq_id)
            if chunk_id not in chunks_to_save:
                chunks_to_save[chunk_id] = {}
            chunks_to_save[chunk_id][seq_id] = self.sequence_buffer[seq_id]
        
        # Save each partial chunk
        for chunk_id in sorted(chunks_to_save.keys()):
            chunk_data = chunks_to_save[chunk_id]
            chunk_meta = self._save_single_chunk(chunk_data, pipeline_config, chunk_id)
            if chunk_meta:
                saved_chunks.append(chunk_meta)
                seq_ids = sorted(chunk_data.keys())
                self.logger.info(
                    f"Saved partial chunk {chunk_id}: sequences {seq_ids[0]}-{seq_ids[-1]} "
                    f"({len(chunk_data)} sequences)"
                )
        
        # Clear the buffer
        self.sequence_buffer.clear()
        
        return saved_chunks


def save_sequence_individually_chunked(chunk_manager: ChunkManager, result_buffer: Dict[int, Dict], 
                                      pipeline_config: Dict, logger) -> List[ChunkMetadata]:
    """
    Save accumulated sequence results, maintaining contiguous chunks.
    
    This function processes sequences individually and only saves complete,
    contiguous chunks of the specified size. This ensures consistent chunk
    sizes and optimal loading performance.
    
    Args:
        chunk_manager: ChunkManager instance for handling storage
        result_buffer: Dictionary of sequence_id -> result data
        pipeline_config: Pipeline configuration for determining what to save
        logger: Logger instance for progress reporting
    
    Returns:
        List of ChunkMetadata objects for any chunks that were saved
    """
    if not result_buffer:
        logger.debug("Empty result buffer, skipping chunk save")
        return []
    
    saved_chunks = []
    
    # Process each sequence individually
    for seq_id, result in result_buffer.items():
        # Convert GPU tensors to CPU numpy arrays for storage
        processed_result = {}
        
        # Always save contextual embeddings and basic data
        processed_result['sequence'] = result['sequence']
        processed_result['span_candidates'] = result['span_candidates']
        processed_result['contextual_embeddings'] = result['contextual_embeddings'].detach().cpu().numpy()
        
        # Conditionally save other components based on config
        output_config = pipeline_config.get('output', {})
        
        if output_config.get('save_seed_embeddings', False):
            processed_result['seed_embeddings'] = result['seed_embeddings'].detach().cpu().numpy()
        
        if output_config.get('save_soft_probabilities', False):
            processed_result['soft_probabilities'] = result['soft_probabilities'].detach().cpu().numpy()
        
        # Add to buffer and get any completed chunks
        completed_chunks = chunk_manager.add_sequence_to_buffer(seq_id, processed_result, pipeline_config)
        saved_chunks.extend(completed_chunks)
    
    return saved_chunks


def save_sequence_results_chunked(chunk_manager: ChunkManager, result_buffer: Dict[int, Dict], 
                                  pipeline_config: Dict, logger) -> Optional[ChunkMetadata]:
    """
    Save accumulated sequence results as a chunk when buffer is full.
    
    Legacy version that maintains original behavior for backwards compatibility.
    For contiguous chunk management, use save_sequence_individually_chunked.
    
    Args:
        chunk_manager: ChunkManager instance for handling storage
        result_buffer: Dictionary of sequence_id -> result data
        pipeline_config: Pipeline configuration for determining what to save
        logger: Logger instance for progress reporting
    
    Returns:
        ChunkMetadata if a chunk was saved, None otherwise
    """
    if not result_buffer:
        logger.debug("Empty result buffer, skipping chunk save")
        return None
    
    # Convert GPU tensors to CPU numpy arrays for storage
    processed_buffer = {}
    total_size_mb = 0
    
    for seq_id, result in result_buffer.items():
        processed_result = {}
        
        # Always save contextual embeddings and basic data
        processed_result['sequence'] = result['sequence']
        processed_result['span_candidates'] = result['span_candidates']
        processed_result['contextual_embeddings'] = result['contextual_embeddings'].detach().cpu().numpy()
        
        # Estimate size for logging
        total_size_mb += result['contextual_embeddings'].numel() * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        # Conditionally save other components based on config
        output_config = pipeline_config.get('output', {})
        
        if output_config.get('save_seed_embeddings', False):
            processed_result['seed_embeddings'] = result['seed_embeddings'].detach().cpu().numpy()
            total_size_mb += result['seed_embeddings'].numel() * 4 / (1024 * 1024)
        
        if output_config.get('save_soft_probabilities', False):
            processed_result['soft_probabilities'] = result['soft_probabilities'].detach().cpu().numpy()
            total_size_mb += result['soft_probabilities'].numel() * 4 / (1024 * 1024)
        
        processed_buffer[seq_id] = processed_result
    
    # Save the chunk
    chunk_meta = chunk_manager.save_chunk(processed_buffer, pipeline_config)
    
    if chunk_meta:
        seq_ids = sorted(processed_buffer.keys())
        logger.info(f"Saved chunk {chunk_meta.chunk_id}: sequences {seq_ids[0]}-{seq_ids[-1]} "
                   f"({chunk_meta.sequence_count} sequences, {chunk_meta.file_size_mb:.1f}MB)")
    
    return chunk_meta


def save_sequence_results_chunked_legacy(chunk_manager: ChunkManager, result_buffer: Dict[int, Dict], 
                                  pipeline_config: Dict, logger) -> Optional[ChunkMetadata]:
    """
    Legacy version that saves sequences as they arrive (may create uneven chunks).
    
    This is the original implementation kept for backwards compatibility.
    Use save_sequence_results_chunked for better chunk management.
    """
    if not result_buffer:
        logger.debug("Empty result buffer, skipping chunk save")
        return None
    
    # Convert GPU tensors to CPU numpy arrays for storage
    processed_buffer = {}
    total_size_mb = 0
    
    for seq_id, result in result_buffer.items():
        processed_result = {}
        
        # Always save contextual embeddings and basic data
        processed_result['sequence'] = result['sequence']
        processed_result['span_candidates'] = result['span_candidates']
        processed_result['contextual_embeddings'] = result['contextual_embeddings'].detach().cpu().numpy()
        
        # Estimate size for logging
        total_size_mb += result['contextual_embeddings'].numel() * 4 / (1024 * 1024)  # float32 = 4 bytes
        
        # Conditionally save other components based on config
        output_config = pipeline_config.get('output', {})
        
        if output_config.get('save_seed_embeddings', False):
            processed_result['seed_embeddings'] = result['seed_embeddings'].detach().cpu().numpy()
            total_size_mb += result['seed_embeddings'].numel() * 4 / (1024 * 1024)
        
        if output_config.get('save_soft_probabilities', False):
            processed_result['soft_probabilities'] = result['soft_probabilities'].detach().cpu().numpy()
            total_size_mb += result['soft_probabilities'].numel() * 4 / (1024 * 1024)
        
        processed_buffer[seq_id] = processed_result
    
    # Save the chunk
    chunk_meta = chunk_manager.save_chunk(processed_buffer, pipeline_config)
    
    if chunk_meta:
        seq_ids = sorted(processed_buffer.keys())
        logger.info(f"Saved chunk {chunk_meta.chunk_id}: sequences {seq_ids[0]}-{seq_ids[-1]} "
                   f"({chunk_meta.sequence_count} sequences, {chunk_meta.file_size_mb:.1f}MB)")
    
    return chunk_meta
