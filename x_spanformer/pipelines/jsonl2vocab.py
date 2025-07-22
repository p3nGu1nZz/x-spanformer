#!/usr/bin/env python3
"""
jsonl2vocab.py

Recursively load all .jsonl under --in, build a hybrid Unigram‐LM
vocabulary (Sec.3.1), and emit final vocab.jsonl plus per‐stage outputs
under --out.

This implementation follows the mathematical formulation in Section 3.1
of the X-Spanformer paper, including proper OOV calculation based on
uncovered codepoint positions and adaptive pruning criteria.
"""
import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional

import yaml

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.vocab import VocabStats
from x_spanformer.pipelines.shared.text_processor import load_pretrain_records
from x_spanformer.vocab import (
    build_candidate_set,
    validate_vocabulary_completeness,
    induce_vocabulary
)
from x_spanformer.vocab.vocab_logging import setup_vocab_logging, get_vocab_logger

# Module-level logger that gets configured in main()
logger = None


def get_logger() -> logging.Logger:
    """Get the module logger, creating a basic one if none exists."""
    global logger
    if logger is None:
        logger = get_vocab_logger('jsonl2vocab')
    return logger


def parse_args():
    p = argparse.ArgumentParser(
        prog="jsonl2vocab",
        description="Build hybrid Unigram‐LM vocab from JSONL directory"
    )
    p.add_argument(
        "-i", "--in",
        dest="indir", type=Path, required=True,
        help="Input directory; recursively search for *.jsonl"
    )
    p.add_argument(
        "-o", "--out",
        dest="outdir", type=Path, required=True,
        help="Output directory; per‐stage subdirs + final vocab.jsonl"
    )
    p.add_argument(
        "-c", "--config",
        dest="config", type=Path,
        default=Path("config/pipelines/jsonl2vocab.yaml"),
        help="Path to YAML hyperparams (default: config/pipelines/jsonl2vocab.yaml)"
    )
    return p.parse_args()


def load_hparams(path: Path) -> dict:
    """Load hyperparameters from YAML configuration file."""
    logger = get_logger()
    logger.info(f"Loading hyperparameters from: {path}")
    if not path.exists():
        logger.error(f"Config not found: {path}")
        raise FileNotFoundError(path)
    
    with open(path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    
    logger.info("Loaded hyperparameters:")
    for key, value in hparams.items():
        logger.info(f"  {key}: {value}")
    
    return hparams


def find_jsonl_files(indir: Path) -> List[Path]:
    """Find all JSONL files recursively in the input directory."""
    logger = get_logger()
    logger.info(f"Searching for JSONL files in: {indir}")
    
    if not indir.exists():
        logger.error(f"Input directory does not exist: {indir}")
        raise FileNotFoundError(f"Input directory not found: {indir}")
    
    files = list(indir.rglob("*.jsonl"))
    if not files:
        logger.error(f"No JSONL files found under: {indir}")
        raise SystemExit(1)
    
    logger.info(f"Found {len(files)} JSONL files:")
    for i, f in enumerate(files, 1):
        logger.debug(f"  {i:3d}. {f}")
    
    return files


def load_corpus(files: List[Path]) -> List[str]:
    """Load corpus from JSONL files using PretrainRecord schema."""
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("STAGE 1: CORPUS LOADING")
    logger.info("=" * 50)
    
    all_sequences = []
    total_stats = {'total': 0, 'valid': 0, 'discarded': 0, 'invalid': 0, 'too_long': 0}
    
    for f in files:
        logger.info(f"Processing file: {f}")
        
        # Use shared function for consistent loading
        sequences, stats = load_pretrain_records(str(f))
        all_sequences.extend(sequences)
        
        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += stats[key]
        
        logger.info(f"  File summary: {stats['valid']} valid, {stats['invalid']} invalid records")
    
    # Calculate corpus statistics
    total_chars = sum(len(seg) for seg in all_sequences)
    avg_segment_length = total_chars / len(all_sequences) if all_sequences else 0
    
    logger.info("-" * 50)
    logger.info("CORPUS LOADING SUMMARY:")
    logger.info(f"  Total input records: {total_stats['total']}")
    logger.info(f"  Valid segments: {total_stats['valid']}")
    logger.info(f"  Invalid records: {total_stats['invalid']}")
    logger.info(f"  Discarded records: {total_stats['discarded']}")
    logger.info(f"  Success rate: {total_stats['valid']/total_stats['total']*100:.1f}%")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Average segment length: {avg_segment_length:.1f} chars")
    logger.info("-" * 50)
    
    return all_sequences


def build_candidate_set_with_output(corpus: List[str], L_max: int, M: int, out: Path) -> Tuple[List[str], Counter]:
    """
    Build candidate set and save output files for inspection.
    
    Following Section 3.1 of the paper:
    U_0 = {top M substrings} ∪ {all single codepoints}
    
    This extracts all substrings up to length L_max, retains the top M by frequency,
    and includes all individual codepoints to ensure base coverage.
    
    Args:
        corpus: List of text segments from JSONL files
        L_max: Maximum substring length (paper: L_max)
        M: Number of top multi-character substrings to keep (paper: M)
        out: Output directory for intermediate results
        
    Returns:
        Tuple of (candidate_vocabulary_U_0, frequency_counter)
    """
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("STAGE 2: CANDIDATE SET GENERATION")
    logger.info("=" * 50)
    logger.info(f"Parameters: L_max={L_max}, M_candidates={M}")
    
    freq_dir = out / "full_freq"
    freq_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    
    # Use the core function
    logger.info("Extracting candidate substrings from corpus...")
    U_0, freq = build_candidate_set(corpus, L_max, M)
    
    extraction_time = time.time() - start_time
    logger.info(f"Candidate extraction completed in {extraction_time:.2f} seconds")
    
    # Save full frequency distribution
    logger.info("Saving frequency distribution...")
    with open(freq_dir / "full_freq.json", "w", encoding="utf-8") as f:
        json.dump(freq.most_common(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved full frequency distribution: {len(freq)} unique substrings")

    # Build candidate set U_0
    cand_dir = out / "candidates"
    cand_dir.mkdir(exist_ok=True)

    # Count different types
    multi_char_count = len([u for u in U_0 if len(u) > 1])
    single_char_count = len([u for u in U_0 if len(u) == 1])
    
    logger.info("-" * 50)
    logger.info("CANDIDATE SET SUMMARY:")
    logger.info(f"  Total candidates: {len(U_0)}")
    logger.info(f"  Multi-character substrings: {multi_char_count}")
    logger.info(f"  Single characters: {single_char_count}")
    logger.info(f"  Extraction time: {extraction_time:.2f}s")
    
    # Separate candidates by type for better analysis
    single_char_candidates = [(u, freq[u]) for u in U_0 if len(u) == 1]
    multi_char_candidates = [(u, freq[u]) for u in U_0 if len(u) > 1]
    
    # Sort by frequency
    single_char_candidates.sort(key=lambda x: x[1], reverse=True)
    multi_char_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Log top single characters
    logger.info("Top 10 most frequent single characters:")
    for i, (char, count) in enumerate(single_char_candidates[:10], 1):
        display_char = repr(char)
        logger.info(f"  {i:2d}. {display_char:<8} (freq: {count:,})")
    
    # Log top multi-character substrings
    logger.info("Top 10 most frequent multi-character substrings:")
    for i, (substring, count) in enumerate(multi_char_candidates[:10], 1):
        display_substring = repr(substring)
        logger.info(f"  {i:2d}. {display_substring:<20} (freq: {count:,})")
    
    logger.info("-" * 50)
    
    # Save candidates for inspection
    logger.info("Saving candidates list...")
    with open(cand_dir / "candidates.txt", "w", encoding="utf-8") as f:
        for u in U_0:
            # Replace newlines with visible symbol for readability
            display_u = u.replace("\n", "⏎")
            f.write(display_u + "\n")

    return U_0, freq


def save_vocab(path: Path, V: List[str], p_u: Dict[str, float], stats: Dict) -> None:
    """Save the final vocabulary to JSONL format using the schema structure."""
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("STAGE 4: VOCABULARY SERIALIZATION")
    logger.info("=" * 50)
    
    logger.info(f"Saving final vocabulary to: {path}")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort vocabulary pieces by probability (descending order - most probable first)
    logger.info("Sorting vocabulary by probability (most probable first)...")
    vocab_sorted = sorted(V, key=lambda u: p_u[u], reverse=True)
    
    vocab_size = len(vocab_sorted)
    logger.info(f"Writing {vocab_size} vocabulary pieces (sorted by probability)...")
    
    # Log top few pieces for verification
    logger.info("Top 5 most probable pieces:")
    for i, u in enumerate(vocab_sorted[:5], 1):
        display_u = repr(u)
        prob = p_u[u]
        logger.info(f"  {i}. {display_u:<15} (prob: {prob:.8f})")
    
    with open(path, "w", encoding="utf-8") as f:
        for i, u in enumerate(vocab_sorted):
            # Create vocabulary piece record following the schema
            rec = {"piece": u, "prob": p_u[u]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            if (i + 1) % 1000 == 0:
                logger.debug(f"  Written {i + 1}/{vocab_size} pieces")
    
    logger.info(f"Vocabulary saved successfully: {vocab_size} pieces (sorted by probability)")
    
    # Save vocabulary statistics using VocabStats schema
    stats_path = path.parent / "vocab_stats.json"
    logger.info(f"Saving vocabulary statistics to: {stats_path}")
    
    vocab_stats = VocabStats(
        total_pieces=stats.get("total_pieces", len(V)),
        baseline_ppl=stats.get("baseline_ppl", 0.0),
        final_ppl=stats.get("final_ppl", 0.0), 
        oov_rate=stats.get("oov_rate", 0.0),
        em_iterations=stats.get("em_iterations", 0),
        pruned_pieces=stats.get("pruned_pieces", 0)
    )
    
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(vocab_stats.model_dump_json(indent=2))
    
    logger.info("-" * 50)
    logger.info("FINAL VOCABULARY STATISTICS:")
    logger.info(f"  Total pieces: {vocab_stats.total_pieces}")
    logger.info(f"  Baseline perplexity: {vocab_stats.baseline_ppl:.4f}")
    logger.info(f"  Final perplexity: {vocab_stats.final_ppl:.4f}")
    logger.info(f"  OOV rate: {vocab_stats.oov_rate:.6f}")
    logger.info(f"  EM iterations: {vocab_stats.em_iterations}")
    logger.info(f"  Pruned pieces: {vocab_stats.pruned_pieces}")
    
    if vocab_stats.baseline_ppl > 0:
        ppl_improvement = vocab_stats.baseline_ppl - vocab_stats.final_ppl
        ppl_improvement_pct = (ppl_improvement / vocab_stats.baseline_ppl) * 100
        logger.info(f"  Perplexity improvement: {ppl_improvement:.4f} ({ppl_improvement_pct:.1f}%)")
    
    logger.info("-" * 50)


def main():
    args = parse_args()
    
    # Setup comprehensive logging first and set the module-level logger
    global logger
    logger = setup_vocab_logging(args.outdir, 'jsonl2vocab')
    
    # Log command line arguments
    logger.info("COMMAND LINE ARGUMENTS:")
    logger.info(f"  Input directory: {args.indir}")
    logger.info(f"  Output directory: {args.outdir}")
    logger.info(f"  Config file: {args.config}")
    logger.info("-" * 80)
    
    # Load hyperparameters
    h = load_hparams(args.config)
    
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    # Pipeline stages following Algorithm 1 from Section 3.1 of the paper:
    # "Adaptive Unigram-LM Vocabulary Induction"
    logger.info("[bold cyan]═══ X-Spanformer JSONL2VOCAB Pipeline ═══[/bold cyan]")
    logger.info("[green]✔ Initializing vocabulary induction pipeline[/green]")
    
    pipeline_start_time = time.time()
    
    # Algorithm Step 1: Extract candidate substrings up to length L_max from corpus
    files = find_jsonl_files(args.indir)
    corpus = load_corpus(files)
    
    # Build initial vocabulary U_0 with frequency-based candidate selection
    U_0, freq = build_candidate_set_with_output(corpus, h["L_max"], h["M_candidates"], out)
    
    # Validate vocabulary completeness before proceeding
    logger.info("Validating vocabulary completeness...")
    validate_vocabulary_completeness(corpus, U_0)
    logger.info("Vocabulary completeness validation passed")
    
    # Algorithm Steps 2-19: EM-based vocabulary induction with adaptive pruning
    # - Initialize piece probabilities p^(0)(u) ∝ freq(u) 
    # - Compute baseline perplexity PPL^(0) via Viterbi decoding
    # - Alternate E-step (Viterbi segmentation) and M-step (probability updates)
    # - Apply adaptive pruning with PPL and OOV constraints
    logger.info("=" * 50)
    logger.info("STAGE 3: EM-BASED VOCABULARY INDUCTION")
    logger.info("=" * 50)
    
    V_final, p_final, stats = induce_vocabulary(corpus, U_0, freq, h, out)
    
    logger.info("EM algorithm completed successfully")
    
    # Save final vocabulary with statistics following VocabStats schema
    save_vocab(out / "vocab.jsonl", V_final, p_final, stats)

    # Calculate total pipeline time
    pipeline_time = time.time() - pipeline_start_time
    
    # Final summary
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {pipeline_time:.2f} seconds")
    logger.info(f"Final vocabulary size: {len(V_final)} pieces")
    logger.info(f"Output files saved to: {out}")
    logger.info("=" * 80)

    logger.info(f"[bold green]✅ Vocabulary induction complete! → {out / 'vocab.jsonl'}[/bold green]")
    logger.info(f"[dim]Final vocabulary size: {len(V_final)} pieces[/dim]")
    logger.info(f"[dim]Baseline PPL: {stats.get('baseline_ppl', 'N/A'):.2f}, Final PPL: {stats.get('final_ppl', 'N/A'):.2f}[/dim]")
    logger.info(f"[dim]OOV rate: {stats.get('oov_rate', 0.0):.4f}, EM iterations: {stats.get('em_iterations', 0)}[/dim]")
    logger.info(f"[dim]Pipeline completed in {pipeline_time:.1f}s[/dim]")
    logger.info(f"[green]📋 Detailed logs available at: {out / 'vocab.log'}[/green]")


if __name__ == "__main__":
    main()