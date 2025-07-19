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
import math
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional

import yaml
from rich.console import Console

# Add the parent directory to the path to import schema modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.vocab import VocabStats
from x_spanformer.agents.rich_utils import console
from x_spanformer.vocab import (
    build_candidate_set,
    validate_vocabulary_completeness,
    induce_vocabulary
)

# Use the shared console from rich_utils for consistency


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
    if not path.exists():
        console.log(f"[red]Config not found:[/red] {path}")
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_jsonl_files(indir: Path) -> List[Path]:
    """Find all JSONL files recursively in the input directory."""
    if not indir.exists():
        console.log(f"[red]Input directory does not exist:[/red] {indir}")
        raise FileNotFoundError(f"Input directory not found: {indir}")
    
    files = list(indir.rglob("*.jsonl"))
    if not files:
        console.log(f"[red]No JSONL files found under:[/red] {indir}")
        raise SystemExit(1)
    
    console.log(f"[green]Found {len(files)} JSONL files under {indir}[/green]")
    return files


def load_corpus(files: List[Path]) -> List[str]:
    """Load corpus from JSONL files using PretrainRecord schema."""
    segments = []
    n = 0
    for f in files:
        console.log(f"Loading segments from: {f}")
        with open(f, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                try:
                    rec_dict = json.loads(line)
                    rec = PretrainRecord(**rec_dict)
                    # Use raw field as specified in the paper
                    txt = rec.raw.strip()
                    if txt:
                        segments.append(txt)
                        n += 1
                except (json.JSONDecodeError, ValueError) as e:
                    console.log(f"[yellow]Warning: Skipping invalid record at line {line_num} in {f}: {e}[/yellow]")
                    continue
    console.log(f"[green]Total segments loaded: {n}[/green]")
    return segments


def build_candidate_set_with_output(corpus: List[str], L_max: int, M: int, out: Path) -> Tuple[List[str], Counter]:
    """
    Build candidate set and save output files for inspection.
    Wrapper around the core build_candidate_set function.
    """
    freq_dir = out / "full_freq"
    freq_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"→ Counting substrings (L_max={L_max})")
    
    # Use the core function
    U_0, freq = build_candidate_set(corpus, L_max, M)
    
    # Save full frequency distribution
    with open(freq_dir / "full_freq.json", "w", encoding="utf-8") as f:
        json.dump(freq.most_common(), f, ensure_ascii=False, indent=2)
    console.log(f"[green]Wrote full_freq.json ({len(freq)} unique substrings)[/green]")

    # Build candidate set U_0
    cand_dir = out / "candidates"
    cand_dir.mkdir(exist_ok=True)

    # Count different types
    multi_char_count = len([u for u in U_0 if len(u) > 1])
    single_char_count = len([u for u in U_0 if len(u) == 1])
    
    console.log(f"[green]Selected {len(U_0)} candidates ({multi_char_count} multi-char + {single_char_count} single)[/green]")

    # Save candidates for inspection
    with open(cand_dir / "candidates.txt", "w", encoding="utf-8") as f:
        for u in U_0:
            # Replace newlines with visible symbol for readability
            display_u = u.replace("\n", "⏎")
            f.write(display_u + "\n")

    return U_0, freq

def save_vocab(path: Path, V: List[str], p_u: Dict[str, float]) -> None:
    """Save the final vocabulary to JSONL format using the schema structure."""
    console.log(f"Saving vocab → {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in V:
            # Create vocabulary piece record following the schema
            rec = {"piece": u, "prob": p_u[u]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    h = load_hparams(args.config)
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    # Pipeline stages following the paper's Algorithm 1
    console.log("[bold cyan]═══ X-Spanformer JSONL2VOCAB Pipeline ═══[/bold cyan]")
    console.log("[green]✔ Initializing vocabulary induction pipeline[/green]")
    
    # Stage 1: Discover and load corpus
    files = find_jsonl_files(args.indir)
    corpus = load_corpus(files)
    
    # Stage 2: Build candidate set U_0
    U_0, freq = build_candidate_set_with_output(corpus, h["L_max"], h["M_candidates"], out)
    
    # Validate vocabulary completeness before proceeding
    validate_vocabulary_completeness(corpus, U_0)
    
    # Stage 3: EM-based vocabulary induction with adaptive pruning
    V_final, p_final, stats = induce_vocabulary(corpus, U_0, freq, h, out)
    
    # Stage 4: Save final vocabulary
    save_vocab(out / "vocab.jsonl", V_final, p_final)

    console.log(f"[bold green]✅ Vocabulary induction complete! → {out / 'vocab.jsonl'}[/bold green]")
    console.log(f"[dim]Final vocabulary size: {len(V_final)} pieces[/dim]")


if __name__ == "__main__":
    main()