"""
Shared JSONL processing utilities for all X-Spanformer pipelines.

This module centralizes JSONL file handling, dataset creation, and corpus 
management functionality used across pdf2jsonl, jsonl2vocab, vocab2embedding,
and repo2jsonl pipelines.

Key Functions:
- load_pretrain_records: Load and validate PretrainRecord format files
- save_consolidated_corpus: Create consolidated corpus.jsonl for downstream processing
- discover_jsonl_files: Find JSONL files recursively in directories
- validate_jsonl_schema: Validate JSONL records against schemas
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator

from x_spanformer.agents.rich_utils import console

logger = logging.getLogger(__name__)


def discover_jsonl_files(input_path: Path, recursive: bool = True) -> List[Path]:
    """
    Discover JSONL files in a directory or return single file if path is a file.
    
    Args:
        input_path: Path to directory or single JSONL file
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of JSONL file paths
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If no JSONL files found
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.jsonl':
            return [input_path]
        else:
            raise ValueError(f"File is not a JSONL file: {input_path}")
    
    # Directory - search for JSONL files
    if recursive:
        files = list(input_path.rglob("*.jsonl"))
    else:
        files = list(input_path.glob("*.jsonl"))
    
    if not files:
        logger.error(f"No JSONL files found in: {input_path}")
        sys.exit(1)
    
    logger.info(f"Discovered {len(files)} JSONL files in {input_path}")
    return sorted(files)


def load_pretrain_records(file_path: str, max_length: Optional[int] = None) -> Tuple[List[str], Dict]:
    """
    Load PretrainRecord format from a single JSONL file.
    
    Args:
        file_path: Path to JSONL file with PretrainRecord format
        max_length: Optional maximum sequence length filter
        
    Returns:
        Tuple of (sequences, statistics)
    """
    from x_spanformer.schema.pretrain_record import PretrainRecord
    
    sequences = []
    stats = {'total': 0, 'valid': 0, 'discarded': 0, 'invalid': 0, 'too_long': 0}
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading PretrainRecord sequences from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                stats['total'] += 1
                record_data = json.loads(line)
                
                # Validate using PretrainRecord schema
                try:
                    record = PretrainRecord(**record_data)
                except Exception as e:
                    logger.debug(f"Line {line_num}: Schema validation failed: {e}")
                    stats['invalid'] += 1
                    continue
                
                # Skip explicitly discarded sequences
                if record.meta and hasattr(record.meta, 'status') and record.meta.status == 'discard':
                    stats['discarded'] += 1
                    continue
                
                # Extract and validate sequence
                sequence = record.raw.strip()
                if not sequence:
                    stats['invalid'] += 1
                    continue
                
                # Apply length filter
                if max_length and len(sequence) > max_length:
                    stats['too_long'] += 1
                    continue
                
                sequences.append(sequence)
                stats['valid'] += 1
                
            except json.JSONDecodeError as e:
                logger.debug(f"Line {line_num}: JSON decode error: {e}")
                stats['invalid'] += 1
            except Exception as e:
                logger.debug(f"Line {line_num}: Unexpected error: {e}")
                stats['invalid'] += 1
    
    logger.info(f"Loaded {stats['valid']} valid sequences from {stats['total']} total records")
    if stats['discarded'] > 0:
        logger.info(f"Skipped {stats['discarded']} explicitly discarded records")
    if stats['too_long'] > 0:
        logger.info(f"Skipped {stats['too_long']} sequences longer than {max_length}")
    if stats['invalid'] > 0:
        logger.warning(f"Failed to process {stats['invalid']} invalid records")
    
    return sequences, stats


def load_corpus_from_multiple_files(file_paths: List[Path], max_length: Optional[int] = None) -> Tuple[List[str], Dict]:
    """
    Load corpus from multiple JSONL files and consolidate statistics.
    
    Args:
        file_paths: List of JSONL file paths
        max_length: Optional maximum sequence length filter
        
    Returns:
        Tuple of (all_sequences, consolidated_statistics)
    """
    all_sequences = []
    total_stats = {'total': 0, 'valid': 0, 'discarded': 0, 'invalid': 0, 'too_long': 0}
    
    logger.info(f"Loading corpus from {len(file_paths)} JSONL files")
    
    for file_path in file_paths:
        logger.info(f"Processing: {file_path}")
        
        # Load sequences from this file
        sequences, stats = load_pretrain_records(str(file_path), max_length)
        all_sequences.extend(sequences)
        
        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += stats[key]
        
        logger.debug(f"  File summary: {stats['valid']} valid, {stats['invalid']} invalid")
    
    # Calculate overall corpus statistics
    total_chars = sum(len(seq) for seq in all_sequences)
    avg_length = total_chars / len(all_sequences) if all_sequences else 0
    
    logger.info(f"Corpus loading complete:")
    logger.info(f"  Total files: {len(file_paths)}")
    logger.info(f"  Valid sequences: {total_stats['valid']:,}")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average sequence length: {avg_length:.1f}")
    
    return all_sequences, total_stats


def save_consolidated_corpus(sequences: List[str], output_path: Path, 
                           filename: str = "corpus.jsonl",
                           source_info: str = "consolidated") -> Path:
    """
    Save consolidated corpus in PretrainRecord format for downstream processing.
    
    Args:
        sequences: List of text sequences to save
        output_path: Output directory
        filename: Name of the output file (default: corpus.jsonl)
        source_info: Source information for metadata
        
    Returns:
        Path to the created corpus file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    corpus_path = output_path / filename
    
    logger.info(f"Saving consolidated corpus: {corpus_path}")
    logger.info(f"  Sequences: {len(sequences):,}")
    
    timestamp = datetime.now().isoformat()
    
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, sequence in enumerate(sequences, 1):
            # Create PretrainRecord-compatible format
            record = {
                "raw": sequence,
                "type": "mixed",
                "id": {"id": f"corpus-seq-{i:08d}"},
                "meta": {
                    "status": "keep",
                    "extracted_by": source_info,
                    "timestamp": timestamp,
                    "sequence_number": i,
                    "source": "consolidated_corpus"
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Consolidated corpus saved: {corpus_path} ({len(sequences)} sequences)")
    return corpus_path


def validate_jsonl_file(file_path: Path, schema_class=None) -> Dict:
    """
    Validate a JSONL file against a schema (optional) and return statistics.
    
    Args:
        file_path: Path to JSONL file
        schema_class: Optional Pydantic model class for validation
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {'total_lines': 0, 'valid_json': 0, 'valid_schema': 0, 'errors': []}
    
    logger.info(f"Validating JSONL file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1
            
            # Test JSON parsing
            try:
                data = json.loads(line)
                stats['valid_json'] += 1
                
                # Test schema validation if provided
                if schema_class:
                    try:
                        schema_class(**data)
                        stats['valid_schema'] += 1
                    except Exception as e:
                        stats['errors'].append(f"Line {line_num}: Schema error: {e}")
                        
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: JSON error: {e}")
    
    success_rate = stats['valid_json'] / stats['total_lines'] * 100 if stats['total_lines'] > 0 else 0
    schema_rate = stats['valid_schema'] / stats['total_lines'] * 100 if schema_class and stats['total_lines'] > 0 else None
    
    logger.info(f"Validation complete:")
    logger.info(f"  Total lines: {stats['total_lines']}")
    logger.info(f"  Valid JSON: {stats['valid_json']} ({success_rate:.1f}%)")
    if schema_rate is not None:
        logger.info(f"  Valid schema: {stats['valid_schema']} ({schema_rate:.1f}%)")
    if stats['errors']:
        logger.warning(f"  Errors: {len(stats['errors'])}")
        for error in stats['errors'][:5]:  # Show first 5 errors
            logger.warning(f"    {error}")
    
    return stats


def stream_jsonl_records(file_path: Path) -> Iterator[Dict]:
    """
    Stream JSONL records one at a time for memory-efficient processing.
    
    Args:
        file_path: Path to JSONL file
        
    Yields:
        Dictionary objects from each JSONL line
    """
    logger.debug(f"Streaming records from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON: {e}")
                continue


def merge_jsonl_files(input_files: List[Path], output_file: Path, 
                     validate_schema: bool = True) -> Dict:
    """
    Merge multiple JSONL files into a single consolidated file.
    
    Args:
        input_files: List of input JSONL files
        output_file: Output merged JSONL file
        validate_schema: Whether to validate PretrainRecord schema
        
    Returns:
        Dictionary with merge statistics
    """
    from x_spanformer.schema.pretrain_record import PretrainRecord
    
    stats = {'input_files': len(input_files), 'total_records': 0, 'valid_records': 0, 'errors': 0}
    
    logger.info(f"Merging {len(input_files)} JSONL files into {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outf:
        for input_file in input_files:
            logger.debug(f"  Processing: {input_file}")
            
            for record in stream_jsonl_records(input_file):
                stats['total_records'] += 1
                
                # Optional schema validation
                if validate_schema:
                    try:
                        PretrainRecord(**record)
                    except Exception as e:
                        logger.debug(f"Schema validation failed: {e}")
                        stats['errors'] += 1
                        continue
                
                # Write record to output
                outf.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['valid_records'] += 1
    
    logger.info(f"Merge complete:")
    logger.info(f"  Input files: {stats['input_files']}")
    logger.info(f"  Total records: {stats['total_records']:,}")
    logger.info(f"  Valid records: {stats['valid_records']:,}")
    if stats['errors'] > 0:
        logger.warning(f"  Errors: {stats['errors']:,}")
    
    return stats


def load_corpus_with_logging(file_paths: List[Path], max_length: Optional[int] = None, 
                           stage_name: str = "CORPUS LOADING") -> List[str]:
    """
    Load corpus from multiple JSONL files with comprehensive logging.
    This is a higher-level function that wraps load_corpus_from_multiple_files
    with detailed logging suitable for pipeline stages.
    
    Args:
        file_paths: List of JSONL file paths
        max_length: Optional maximum sequence length filter
        stage_name: Name of the processing stage for logging
        
    Returns:
        List of text sequences from all files
    """
    logger.info("=" * 50)
    logger.info(f"STAGE: {stage_name}")
    logger.info("=" * 50)
    
    # Use the core function
    all_sequences, total_stats = load_corpus_from_multiple_files(file_paths, max_length)
    
    # Calculate additional statistics for detailed logging
    total_chars = sum(len(seq) for seq in all_sequences)
    avg_segment_length = total_chars / len(all_sequences) if all_sequences else 0
    
    logger.info("-" * 50)
    logger.info(f"{stage_name} SUMMARY:")
    logger.info(f"  Total input files: {len(file_paths)}")
    logger.info(f"  Total input records: {total_stats['total']}")
    logger.info(f"  Valid segments: {total_stats['valid']}")
    logger.info(f"  Invalid records: {total_stats['invalid']}")
    logger.info(f"  Discarded records: {total_stats['discarded']}")
    if total_stats['total'] > 0:
        logger.info(f"  Success rate: {total_stats['valid']/total_stats['total']*100:.1f}%")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average segment length: {avg_segment_length:.1f} chars")
    logger.info("-" * 50)
    
    return all_sequences


def create_sample_dataset(sequences: List[str], output_path: Path, 
                        sample_size: int = 1000, seed: int = 42) -> Path:
    """
    Create a sample dataset from a larger corpus for testing/development.
    
    Args:
        sequences: Source sequences
        output_path: Output directory
        sample_size: Number of sequences to sample
        seed: Random seed for reproducibility
        
    Returns:
        Path to sample dataset file
    """
    import random
    
    random.seed(seed)
    
    # Sample sequences
    sample_sequences = random.sample(sequences, min(sample_size, len(sequences)))
    
    # Save sample corpus
    sample_path = save_consolidated_corpus(
        sample_sequences, 
        output_path, 
        filename="corpus_sample.jsonl",
        source_info="sample_generator"
    )
    
    logger.info(f"Created sample dataset: {sample_path} ({len(sample_sequences)} sequences)")
    return sample_path
