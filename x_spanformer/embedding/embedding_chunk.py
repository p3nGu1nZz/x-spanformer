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
    
    def load_single_sequence(self, seq_id: int) -> Optional[Dict[str, Any]]:
        """
        Load a single sequence from its chunk without loading the entire chunk.
        
        This is much more efficient than load_chunk when you only need one sequence.
        
        Args:
            seq_id: Sequence ID to load
            
        Returns:
            Dictionary containing sequence data, or None if failed
        """
        chunk_id = self.get_chunk_id(seq_id)
        
        if chunk_id not in self.chunks_metadata:
            self.logger.warning(f"Chunk {chunk_id} not found in metadata for sequence {seq_id}")
            return None
        
        chunk_file = self.get_chunk_file_path(chunk_id)
        if not chunk_file.exists():
            self.logger.error(f"Chunk file missing: {chunk_file}")
            return None
        
        try:
            # Load only the metadata first to find the sequence index
            data = np.load(chunk_file, allow_pickle=True)
            seq_ids = data['sequence_ids']
            
            # Find the index of our sequence in the chunk
            seq_index = None
            for i, chunk_seq_id in enumerate(seq_ids):
                if int(chunk_seq_id) == seq_id:
                    seq_index = i
                    break
            
            if seq_index is None:
                self.logger.error(f"Sequence {seq_id} not found in chunk {chunk_id}")
                return None
            
            # Load only the data for this specific sequence
            result = {
                'sequence_id': int(seq_ids[seq_index]),
                'sequence': str(data['sequences'][seq_index]),
                'contextual_embeddings': data['contextual_embeddings'][seq_index],
                'span_candidates': [
                    tuple(span) if isinstance(span, list) else span
                    for span in (
                        data['span_candidates'][seq_index].tolist() 
                        if hasattr(data['span_candidates'][seq_index], 'tolist')
                        else data['span_candidates'][seq_index]
                    )
                ]
            }
            
            # Add optional components if they exist
            if 'seed_embeddings' in data:
                result['seed_embeddings'] = data['seed_embeddings'][seq_index]
            if 'soft_probabilities' in data:
                result['soft_probabilities'] = data['soft_probabilities'][seq_index]
            
            self.logger.debug(f"Loaded single sequence {seq_id} from chunk {chunk_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load sequence {seq_id} from chunk {chunk_id}: {e}")
            return None

    def get_existing_sequences(self) -> Set[int]:
        """
        Get set of all existing sequence IDs across all chunks.
        
        This method loads the actual sequence IDs from chunk files rather than
        relying on metadata ranges, to handle cases where chunks have gaps
        due to missing sequences during processing.
        
        Returns:
            Set of sequence IDs that have been processed and stored
        """
        existing_sequences = set()
        
        for chunk_id, chunk_meta in self.chunks_metadata.items():
            # Load actual sequences from chunk file instead of using metadata range
            try:
                chunk_file = self.get_chunk_file_path(chunk_id)
                if chunk_file.exists():
                    data = np.load(chunk_file, allow_pickle=True)
                    if 'sequence_ids' in data:
                        # Add actual sequence IDs from the file
                        actual_seq_ids = data['sequence_ids'].tolist()
                        existing_sequences.update(actual_seq_ids)
                        self.logger.debug(f"Chunk {chunk_id}: loaded {len(actual_seq_ids)} actual sequences")
                    else:
                        self.logger.warning(f"Chunk {chunk_id}: file missing sequence_ids, using metadata range")
                        # Fallback to metadata range if sequence_ids not found
                        existing_sequences.update(chunk_meta.get_sequence_range())
                else:
                    self.logger.warning(f"Chunk {chunk_id}: file not found, skipping")
            except Exception as e:
                self.logger.error(f"Failed to load sequences from chunk {chunk_id}: {e}")
                # Fallback to metadata range on error
                existing_sequences.update(chunk_meta.get_sequence_range())
        
        self.logger.debug(f"Found {len(existing_sequences)} existing sequences across {len(self.chunks_metadata)} chunks")
        return existing_sequences

    def validate_and_get_missing_sequences(self, total_sequences: int) -> Tuple[Set[int], Dict[int, List[int]]]:
        """
        Validate existing chunks and identify missing sequences.
        
        This method checks each existing chunk for completeness and identifies:
        1. Missing sequences within existing chunks (gaps that need repair)
        2. Missing sequences beyond the last processed sequence (new processing)
        
        Args:
            total_sequences: Total number of sequences in the dataset
            
        Returns:
            Tuple of (all_missing_sequences, chunk_gaps) where:
            - all_missing_sequences: Set of all sequence IDs that need processing
            - chunk_gaps: Dict mapping chunk_id -> list of missing sequence IDs in that chunk
        """
        existing_sequences = self.get_existing_sequences()
        all_sequence_ids = set(range(1, total_sequences + 1))
        all_missing = all_sequence_ids - existing_sequences
        
        # Log detailed chunk verification progress
        total_chunks = len(self.chunks_metadata)
        self.logger.info(f"Verifying {total_chunks} chunks for completeness...")
        
        # Analyze gaps within existing chunks
        chunk_gaps = {}
        verified_count = 0
        
        for chunk_id in sorted(self.chunks_metadata.keys()):
            verified_count += 1
            start_seq_id, end_seq_id = self.get_chunk_range(chunk_id)
            expected_chunk_sequences = set(range(start_seq_id, end_seq_id + 1))
            
            # Load actual sequences from this chunk
            try:
                chunk_file = self.get_chunk_file_path(chunk_id)
                if chunk_file.exists():
                    data = np.load(chunk_file, allow_pickle=True)
                    if 'sequence_ids' in data:
                        actual_sequences = set(data['sequence_ids'].tolist())
                        missing_in_chunk = expected_chunk_sequences - actual_sequences
                        # Only include missing sequences that are within the total dataset range
                        missing_in_chunk = missing_in_chunk & set(range(1, total_sequences + 1))
                        
                        if missing_in_chunk:
                            chunk_gaps[chunk_id] = sorted(missing_in_chunk)
                            self.logger.warning(
                                f"Chunk {chunk_id} verification {verified_count}/{total_chunks}: "
                                f"INCOMPLETE - {len(missing_in_chunk)} missing sequences: {sorted(missing_in_chunk)}"
                            )
                        else:
                            self.logger.info(
                                f"Chunk {chunk_id} verification {verified_count}/{total_chunks}: "
                                f"COMPLETE - {len(actual_sequences)}/{len(expected_chunk_sequences)} sequences"
                            )
                    else:
                        self.logger.error(
                            f"Chunk {chunk_id} verification {verified_count}/{total_chunks}: "
                            f"ERROR - missing sequence_ids array"
                        )
                else:
                    self.logger.error(
                        f"Chunk {chunk_id} verification {verified_count}/{total_chunks}: "
                        f"ERROR - chunk file does not exist: {chunk_file}"
                    )
            except Exception as e:
                self.logger.error(f"Chunk {chunk_id} verification {verified_count}/{total_chunks}: ERROR - {e}")
        
        complete_chunks = total_chunks - len(chunk_gaps)
        self.logger.info(
            f"Chunk verification complete: {complete_chunks}/{total_chunks} complete, "
            f"{len(chunk_gaps)} incomplete, {len(all_missing)} total missing sequences"
        )
        
        return all_missing, chunk_gaps

    def repair_incomplete_chunks(self, chunk_gaps: Dict[int, List[int]], 
                                sequence_processor_func, pipeline_config: Dict) -> bool:
        """
        Repair incomplete chunks by processing missing sequences and resaving complete chunks.
        
        Args:
            chunk_gaps: Dict mapping chunk_id -> list of missing sequence IDs
            sequence_processor_func: Function to process sequences (seq_id, sequence_text) -> result_dict
            pipeline_config: Pipeline configuration for saving
            
        Returns:
            True if all repairs successful, False if any failures
        """
        if not chunk_gaps:
            self.logger.info("No incomplete chunks to repair")
            return True
        
        self.logger.info(f"Repairing {len(chunk_gaps)} incomplete chunks...")
        repair_success = True
        
        for chunk_id, missing_seq_ids in chunk_gaps.items():
            try:
                self.logger.info(f"Repairing chunk {chunk_id}: processing {len(missing_seq_ids)} missing sequences")
                
                # Load existing chunk data
                existing_data = self.load_chunk(chunk_id)
                if existing_data is None:
                    self.logger.error(f"Failed to load existing chunk {chunk_id} for repair")
                    repair_success = False
                    continue
                
                # Process missing sequences
                repaired_data = existing_data.copy()
                for seq_id in missing_seq_ids:
                    try:
                        # Call the sequence processor function
                        result = sequence_processor_func(seq_id)
                        if result is not None:
                            repaired_data[seq_id] = result
                            self.logger.debug(f"Processed missing sequence {seq_id} for chunk {chunk_id}")
                        else:
                            self.logger.error(f"Failed to process sequence {seq_id} for chunk {chunk_id}")
                            repair_success = False
                    except Exception as e:
                        self.logger.error(f"Error processing sequence {seq_id} for chunk {chunk_id}: {e}")
                        repair_success = False
                
                # Save the repaired chunk (overwrites existing)
                chunk_meta = self._save_single_chunk(repaired_data, pipeline_config, chunk_id)
                if chunk_meta:
                    seq_ids = sorted(repaired_data.keys())
                    self.logger.info(
                        f"Repaired chunk {chunk_id}: now contains sequences {seq_ids[0]}-{seq_ids[-1]} "
                        f"({len(repaired_data)} sequences, added {len(missing_seq_ids)} missing)"
                    )
                else:
                    self.logger.error(f"Failed to save repaired chunk {chunk_id}")
                    repair_success = False
                    
            except Exception as e:
                self.logger.error(f"Failed to repair chunk {chunk_id}: {e}")
                repair_success = False
        
        if repair_success:
            self.logger.info("All chunk repairs completed successfully")
        else:
            self.logger.warning("Some chunk repairs failed - data integrity may be compromised")
            
        return repair_success
    
    def final_integrity_check(self, total_sequences: int) -> bool:
        """
        Perform final integrity check after pipeline completion to ensure no sequences are missing.
        
        Args:
            total_sequences: Total number of sequences that should be in the dataset
            
        Returns:
            True if all sequences are present and chunks are complete, False otherwise
        """
        self.logger.info("=" * 60)
        self.logger.info("PERFORMING FINAL INTEGRITY CHECK")
        self.logger.info("=" * 60)
        
        # Get all existing sequences
        existing_sequences = self.get_existing_sequences()
        expected_sequences = set(range(1, total_sequences + 1))
        missing_sequences = expected_sequences - existing_sequences
        
        self.logger.info(f"Expected sequences: {total_sequences} (1-{total_sequences})")
        self.logger.info(f"Found sequences: {len(existing_sequences)}")
        
        if missing_sequences:
            self.logger.error(f"INTEGRITY CHECK FAILED: {len(missing_sequences)} sequences missing!")
            missing_ranges = []
            sorted_missing = sorted(missing_sequences)
            start = sorted_missing[0]
            end = start
            
            for seq_id in sorted_missing[1:]:
                if seq_id == end + 1:
                    end = seq_id
                else:
                    if start == end:
                        missing_ranges.append(str(start))
                    else:
                        missing_ranges.append(f"{start}-{end}")
                    start = end = seq_id
            
            if start == end:
                missing_ranges.append(str(start))
            else:
                missing_ranges.append(f"{start}-{end}")
            
            self.logger.error(f"Missing sequence ranges: {', '.join(missing_ranges)}")
            return False
        
        # Validate chunk completeness
        all_missing, chunk_gaps = self.validate_and_get_missing_sequences(total_sequences)
        
        if chunk_gaps:
            self.logger.error(f"INTEGRITY CHECK FAILED: {len(chunk_gaps)} chunks are incomplete!")
            for chunk_id, missing_in_chunk in chunk_gaps.items():
                self.logger.error(f"  Chunk {chunk_id}: missing {len(missing_in_chunk)} sequences: {missing_in_chunk}")
            return False
        
        # All checks passed
        total_chunks = len(self.chunks_metadata)
        self.logger.info(f"INTEGRITY CHECK PASSED: All {total_sequences} sequences present in {total_chunks} complete chunks")
        
        # Log chunk distribution
        sequence_counts = {}
        for chunk_id in sorted(self.chunks_metadata.keys()):
            try:
                chunk_file = self.get_chunk_file_path(chunk_id)
                if chunk_file.exists():
                    data = np.load(chunk_file, allow_pickle=True)
                    if 'sequence_ids' in data:
                        sequence_counts[chunk_id] = len(data['sequence_ids'])
            except Exception as e:
                self.logger.warning(f"Could not count sequences in chunk {chunk_id}: {e}")
        
        total_counted = sum(sequence_counts.values())
        self.logger.info(f"Sequence distribution: {total_counted} sequences across {len(sequence_counts)} chunks")
        if len(sequence_counts) <= 10:  # Show distribution for small number of chunks
            for chunk_id in sorted(sequence_counts.keys()):
                count = sequence_counts[chunk_id]
                start_seq, end_seq = self.get_chunk_range(chunk_id)
                self.logger.info(f"  Chunk {chunk_id}: {count} sequences ({start_seq}-{end_seq})")
        
        self.logger.info("=" * 60)
        return True
    
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
            
            # Handle case where chunk already exists (incomplete chunk continuation)
            if chunk_id in self.chunks_metadata:
                # This chunk already exists - we're completing it
                existing_chunk = self.chunks_metadata[chunk_id]
                existing_sequences = set(existing_chunk.get_sequence_range())
                
                # Check if we now have all missing sequences for this chunk
                missing_sequences = expected_sequences - existing_sequences
                buffered_missing = buffered_for_chunk & missing_sequences
                
                if len(buffered_missing) == len(missing_sequences):
                    # We have all missing sequences - complete the chunk
                    # Load existing chunk data
                    existing_data = self.load_chunk(chunk_id)
                    if existing_data is None:
                        self.logger.error(f"Failed to load existing chunk {chunk_id} for completion")
                        break
                    
                    # Merge with new buffered data
                    chunk_data = existing_data.copy()
                    for seq_id in buffered_missing:
                        chunk_data[seq_id] = self.sequence_buffer[seq_id]
                    
                    # Save the completed chunk (will overwrite existing)
                    chunk_meta = self._save_single_chunk(chunk_data, pipeline_config, chunk_id)
                    if chunk_meta:
                        saved_chunks.append(chunk_meta)
                    
                    # Remove saved sequences from buffer
                    for seq_id in buffered_missing:
                        self.sequence_buffer.pop(seq_id, None)
                        
                    self.logger.info(
                        f"Completed existing chunk {chunk_id}: sequences {start_seq_id}-{end_seq_id} "
                        f"(added {len(buffered_missing)} sequences, {self.chunk_size} total)"
                    )
                else:
                    # Still missing some sequences for this chunk
                    break
            else:
                # New chunk - use original logic
                if len(buffered_for_chunk) == self.chunk_size:
                    # We have a complete new chunk, save it
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
