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


def build_candidate_set(corpus: List[str], L_max: int, M: int, out: Path) -> Tuple[List[str], Counter]:
    """
    Build the initial candidate set U_0 following the paper's formulation.
    
    U_0 = {top M substrings} ∪ {all single codepoints}
    """
    freq_dir = out / "full_freq"
    freq_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"→ Counting substrings (L_max={L_max})")
    freq = Counter()
    
    for x in corpus:
        T = len(x)
        # Count all substrings up to L_max
        for i in range(T):
            # Single characters
            freq[x[i]] += 1
            # Multi-character substrings
            for length in range(2, min(L_max, T - i) + 1):
                freq[x[i:i + length]] += 1

    # Save full frequency distribution
    with open(freq_dir / "full_freq.json", "w", encoding="utf-8") as f:
        json.dump(freq.most_common(), f, ensure_ascii=False, indent=2)
    console.log(f"[green]Wrote full_freq.json ({len(freq)} unique substrings)[/green]")

    # Build candidate set U_0
    cand_dir = out / "candidates"
    cand_dir.mkdir(exist_ok=True)

    # Top M multi-character substrings
    multi_char_substrings = [u for u, _ in freq.most_common() if len(u) > 1][:M]
    
    # All single codepoints
    single_codepoints = list({u for u in freq if len(u) == 1})
    
    # Combine to form U_0
    U_0 = multi_char_substrings + single_codepoints
    
    console.log(f"[green]Selected {len(U_0)} candidates ({len(multi_char_substrings)} multi-char + {len(single_codepoints)} single)[/green]")

    # Save candidates for inspection
    with open(cand_dir / "candidates.txt", "w", encoding="utf-8") as f:
        for u in U_0:
            # Replace newlines with visible symbol for readability
            display_u = u.replace("\n", "⏎")
            f.write(display_u + "\n")

    return U_0, freq


def viterbi_segment(x: str, V: List[str], p_u: Dict[str, float]) -> List[str]:
    """
    Viterbi segmentation following the paper's formulation.
    
    Returns the best segmentation seg*(x) = argmax_seg ∏_{v∈seg} p(v)
    """
    T = len(x)
    dp = [-math.inf] * (T + 1)
    back = [None] * (T + 1)
    dp[0] = 0.0

    # Index vocabulary by first character for efficiency
    by_first = {}
    for u in V:
        if u:  # Skip empty strings
            by_first.setdefault(u[0], []).append(u)

    for i in range(T):
        if dp[i] == -math.inf:
            continue
        for u in by_first.get(x[i], []):
            j = i + len(u)
            if j <= T and x[i:j] == u:
                pu = p_u.get(u, 1e-12)
                sc = dp[i] + math.log(pu)
                if sc > dp[j]:
                    dp[j], back[j] = sc, u

    # Reconstruct path
    seg, ptr = [], T
    while ptr > 0:
        piece = back[ptr] or x[ptr - 1]  # Fallback to single character
        seg.append(piece)
        ptr -= len(piece)
    return list(reversed(seg))


def compute_coverage(x: str, segmentation: List[str]) -> Set[int]:
    """
    Compute the set of codepoint indices covered by the segmentation.
    
    Returns cover_V(x) = {i | i is covered by some piece in segmentation}
    
    Note: This validates that the segmentation properly reconstructs the input.
    """
    covered = set()
    pos = 0
    
    for piece in segmentation:
        if pos + len(piece) <= len(x) and x[pos:pos + len(piece)] == piece:
            covered.update(range(pos, pos + len(piece)))
            pos += len(piece)
        else:
            # This shouldn't happen with proper Viterbi segmentation
            break
    
    return covered


def compute_corpus_coverage(corpus: List[str], V: List[str]) -> float:
    """
    Compute corpus-level coverage as the percentage of codepoints covered.
    """
    total_covered = 0
    total_positions = 0
    
    for x in corpus:
        segmentation = viterbi_segment(x, V, {})  # Use uniform probabilities for coverage
        covered = compute_coverage(x, segmentation)
        total_covered += len(covered)
        total_positions += len(x)
    
    return total_covered / total_positions if total_positions > 0 else 0.0
    covered = set()
    pos = 0
    reconstructed = ""
    
    for piece in segmentation:
        # Track covered positions
        for i in range(pos, pos + len(piece)):
            if i < len(x):  # Ensure we don't go beyond string bounds
                covered.add(i)
        reconstructed += piece
        pos += len(piece)
    
    # Validate reconstruction matches original
    if reconstructed != x:
        raise ValueError(f"Segmentation reconstruction failed: '{reconstructed}' != '{x}'")
    
    return covered


def compute_baseline_perplexity(corpus: List[str], V: List[str], p_u: Dict[str, float]) -> float:
    """
    Compute baseline perplexity following the paper's formulation.
    
    PPL^(0) = exp(-1/|X| * Σ_{x∈X} log ∏_{v∈seg*(x)} p^(0)(v))
    
    This is sequence-level perplexity as defined in the paper.
    """
    total_log_likelihood = 0.0
    num_sequences = len(corpus)
    
    for x in corpus:
        segmentation = viterbi_segment(x, V, p_u)
        sequence_log_prob = 0.0
        
        for piece in segmentation:
            prob = p_u.get(piece, 1e-12)
            sequence_log_prob += math.log(prob)
        
        total_log_likelihood += sequence_log_prob
    
    # Paper's baseline perplexity formula
    ppl = math.exp(-total_log_likelihood / num_sequences)
    return ppl


def compute_pruning_perplexity_and_oov(corpus: List[str], V: List[str], p_u: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute pruning perplexity and OOV rate according to the paper's formulation.
    
    PPL' = exp(L'/N_p') where L' is negative log-likelihood, N_p' is total pieces
    OOV' = N_uncov' / N_t where N_uncov' is uncovered positions, N_t is total codepoints
    
    This is piece-level perplexity used during pruning decisions.
    """
    total_pieces = 0  # N_p'
    total_log_prob = 0.0  # L' (will be negative)
    total_codepoints = 0  # N_t
    uncovered_positions = 0  # N_uncov'
    
    for x in corpus:
        segmentation = viterbi_segment(x, V, p_u)
        coverage = compute_coverage(x, segmentation)
        
        # Update counts
        total_pieces += len(segmentation)
        total_codepoints += len(x)
        uncovered_positions += len(x) - len(coverage)
        
        # Update log probability
        for piece in segmentation:
            prob = p_u.get(piece, 1e-12)
            total_log_prob += math.log(prob)
    
    # Compute metrics according to paper
    ppl = math.exp(-total_log_prob / total_pieces) if total_pieces > 0 else float('inf')
    oov_rate = uncovered_positions / total_codepoints if total_codepoints > 0 else 0.0
    
    return ppl, oov_rate


def induce_vocab(corpus: List[str], V: List[str], freq: Counter, h: Dict, out: Path) -> Tuple[List[str], Dict[str, float]]:
    """
    EM-based Unigram LM with adaptive pruning following the paper's Algorithm 1.
    
    Implements the complete algorithm from Section 3.1 including:
    - Proper initialization of piece probabilities
    - EM iterations with Viterbi E-step and frequency-based M-step  
    - Adaptive pruning with PPL and OOV thresholds
    
    **Mathematical Note:** The paper has an inconsistency where baseline perplexity 
    uses sequence-level normalization but pruning uses piece-level normalization.
    This implementation uses piece-level normalization consistently to enable 
    meaningful comparison during pruning decisions.
    """
    prune_dir = out / "pruning"
    prune_dir.mkdir(exist_ok=True)

    # Store initial vocabulary size for statistics
    V_init = V.copy()  # V is the candidate set from build_candidate_set
    
    # Extract hyperparameters
    T_max = h["T_max_iters"]
    eps = h["min_piece_prob"]
    tau_ppl = h["delta_perplexity"]  # τ_ppl in the paper
    delta_oov = h["delta_oov"]       # δ_oov in the paper

    console.log("→ EM‐based Unigram LM & adaptive pruning")
    
    # Initialize piece probabilities: p^(0)(u) = freq(u) / Σ_v freq(v)
    total_freq = sum(freq[u] for u in V)
    p_u = {u: freq[u] / total_freq for u in V}

    # Compute baseline perplexity PPL^(0) 
    # Note: Using piece-level normalization for consistency with pruning comparisons
    # This differs from the paper's sequence-level baseline to enable meaningful comparison
    baseline_ppl, baseline_oov = compute_pruning_perplexity_and_oov(corpus, V, p_u)
    console.log(f"  Baseline PPL: {baseline_ppl:.4f}, OOV: {baseline_oov:.4f}")
    console.log(f"  [yellow]Note: Using piece-level PPL for consistency with pruning comparisons[/yellow]")

    current_ppl = baseline_ppl
    final_iteration = 0  # Track the final iteration count
    
    # EM iterations
    for iteration in range(1, T_max + 1):
        final_iteration = iteration
        console.log(f"  Iter {iteration}/{T_max}")
        
        # E-step: Compute γ^(t)(u|x) via Viterbi segmentation
        counts = Counter()
        for x in corpus:
            segmentation = viterbi_segment(x, V, p_u)
            for piece in segmentation:
                counts[piece] += 1

        # M-step: Update probabilities p^(t+1)(u) 
        total_counts = sum(counts.values())
        if total_counts > 0:
            p_next = {u: counts.get(u, 0) / total_counts for u in V}
        else:
            p_next = p_u.copy()

        # Adaptive pruning: consider removing pieces with p^(t+1)(u) < ε
        candidates_to_prune = [u for u in V if p_next[u] < eps]
        console.log(f"   Pruning candidates: {len(candidates_to_prune)}")
        
        for u in candidates_to_prune:
            # Tentative removal: V' = V \ {u}
            V_prime = [v for v in V if v != u]
            
            # Simulate removal and compute new metrics using pruning formula
            ppl_prime, oov_prime = compute_pruning_perplexity_and_oov(corpus, V_prime, p_next)
            
            # Check pruning criteria from the paper
            ppl_increase = ppl_prime - current_ppl
            if ppl_increase < tau_ppl and oov_prime <= delta_oov:
                console.log(f"    Pruned '{u}' ΔPPL={ppl_increase:.4f}, OOV={oov_prime:.4f}")
                V = V_prime
                current_ppl = ppl_prime

        # Update probabilities for next iteration
        p_u = p_next

    console.log(f"[green]Final vocab size: {len(V)}[/green]")
    
    # Final vocabulary statistics using schema
    final_coverage = compute_corpus_coverage(corpus, V)
    final_ppl, final_oov_rate = compute_pruning_perplexity_and_oov(corpus, V, p_u)
    
    stats = VocabStats(
        total_pieces=len(V),
        baseline_ppl=baseline_ppl,
        final_ppl=final_ppl,
        oov_rate=final_oov_rate,
        em_iterations=final_iteration,
        pruned_pieces=len(V_init) - len(V)
    )
    
    console.print(f"✅ Final vocabulary: {stats.total_pieces} pieces, "
                  f"PPL={stats.final_ppl:.2f}, OOV={stats.oov_rate:.1%}, "
                  f"Coverage={final_coverage:.1%}")
    
    # Save final probabilities
    with open(prune_dir / "final_probs.json", "w", encoding="utf-8") as f:
        json.dump({u: p_u[u] for u in V}, f, ensure_ascii=False, indent=2)

    return V, p_u


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
    U_0, freq = build_candidate_set(corpus, h["L_max"], h["M_candidates"], out)
    
    # Stage 3: EM-based vocabulary induction with adaptive pruning
    V_final, p_final = induce_vocab(corpus, U_0, freq, h, out)
    
    # Stage 4: Save final vocabulary
    save_vocab(out / "vocab.jsonl", V_final, p_final)

    console.log(f"[bold green]✅ Vocabulary induction complete! → {out / 'vocab.jsonl'}[/bold green]")
    console.log(f"[dim]Final vocabulary size: {len(V_final)} pieces[/dim]")


if __name__ == "__main__":
    main()