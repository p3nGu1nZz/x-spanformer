# JSONL2Vocab Pipeline

# JSONL2Vocab Pipeline

Following Algorithm 1 from Section 3.1 of the X-Spanformer paper, this pipeline ingests one or more JSONL files (recursively discovered in an input directory), extracts each record's text, and induces a hybrid Unigram-LM vocabulary via an EM + Viterbi procedure with adaptive pruning. It implements the mathematical formulation described in Section 3.1 of the X-Spanformer paper, writing per-stage artifacts under an output directory and emitting a final vocabulary as JSONL with comprehensive statistics.

---

## Overview

This pipeline implements the **Adaptive Unigram-LM Vocabulary Induction** algorithm with the following stages:

1. **Discover** all `*.jsonl` under `--in` (recursively).  
2. **Load** each record's `raw` field using `PretrainRecord` schema validation via shared utilities.  
3. **Count** all substrings up to length `L_max`; dump `full_freq.json`.  
4. **Select** top `M` substrings + all single codepoints to form U₀; dump `candidates.txt`.  
5. **EM-train** a Unigram LM with Viterbi-based segmentation, adaptively pruning low-probability pieces under perplexity and OOV thresholds; dump `final_probs.json`.  
6. **Save** the final vocabulary (`piece` + `prob`) in `vocab.jsonl` using `VocabPiece` schema.
7. **Output** comprehensive statistics in `vocab_stats.json` using `VocabStats` schema.

---

## Overview

This pipeline implements the **Adaptive Unigram-LM Vocabulary Induction** algorithm with the following stages:

1. **Discover** all `*.jsonl` under `--in` (recursively).  
2. **Load** each record's `raw` field using `PretrainRecord` schema validation.  
3. **Count** all substrings up to length `L_max`; dump `full_freq.json`.  
4. **Select** top `M` substrings + all single codepoints to form U₀; dump `candidates.txt`.  
5. **EM-train** a Unigram LM with Viterbi-based segmentation, adaptively pruning low-probability pieces under perplexity and OOV thresholds; dump `final_probs.json`.  
6. **Save** the final vocabulary (`piece` + `prob`) in `vocab.jsonl` using `VocabPiece` schema.
7. **Output** comprehensive statistics in `vocab_stats.json` using `VocabStats` schema.ine inge## Overview

Following Algorithm 1 from Section 3.1 of the X-Spanformer paper, this pipeline:

1. **Discover** all `*.jsonl` under `--in` (recursively).  
2. **Load** each record's `raw` field using `PretrainRecord` schema validation.  
3. **Count** all substrings up to length `L_max`; dump `full_freq.json`.  
4. **Select** top `M` substrings + all single codepoints to form U₀; dump `candidates.txt`.  
5. **EM-train** a Unigram LM with Viterbi-based segmentation, adaptively pruning low-probability pieces under perplexity and OOV thresholds; dump `final_probs.json`.  
6. **Save** the final vocabulary (`piece` + `prob`) in `vocab.jsonl` using `VocabPiece` schema.or more JSONL files (recursively discovered in an input directory), extracts each record's text, and induces a hybrid Unigram-LM vocabulary via an EM + Viterbi procedure with adaptive pruning. It implements the mathematical formulation described in Section 3.1 of the X-Spanformer paper, writing per-stage artifacts under an output directory and emitting a final vocabulary as JSONL.sonl2vocab Pipeline

This pipeline ingests one or more JSONL files (recursively discovered in an input directory), extracts each record’s text, and induces a hybrid Unigram-LM vocabulary via an EM + Viterbi procedure with adaptive pruning. It writes per-stage artifacts under an output directory and emits a final vocabulary as JSONL.

---

## Table of Contents

- [Overview](#overview)  
- [Prerequisites](#prerequisites)  
- [Configuration](#configuration)  
- [Input Format](#input-format)  
- [Pipeline Stages & Outputs](#pipeline-stages--outputs)  
- [Mathematical Foundation](#mathematical-foundation)  
- [CLI Usage](#cli-usage)  
- [Example](#example)  
- [Hyperparameter Reference](#hyperparameter-reference)  

---

## Overview

1. **Discover** all `*.jsonl` under `--in` (recursively).  
2. **Load** each record’s `raw` field (fallback to `text`) as the input string.  
3. **Count** all substrings up to length `L_max`; dump `full_freq.json`.  
4. **Select** top `M` substrings + all single codepoints; dump `candidates.txt`.  
5. **EM-train** a Unigram LM with Viterbi-based segmentation, adaptively pruning low-probability pieces under perplexity and OOV thresholds; dump `final_probs.json`.  
6. **Save** the final vocabulary (`piece` + `prob`) in `vocab.jsonl`.

---

## Prerequisites

All dependencies are assumed to be available as specified in `pyproject.toml`:

- Python 3.8+  
- `PyYAML` - Configuration file parsing  
- `rich` - Enhanced console output and progress tracking  
- `pydantic` - Schema validation for `PretrainRecord`, `VocabPiece`, and `VocabStats`
- Standard library: `argparse`, `json`, `math`, `pathlib`, `collections`, `typing`  

The pipeline uses shared text processing utilities from `x_spanformer.pipelines.shared.text_processor` for consistent corpus loading across all pipelines.

---

## Configuration

Default hyperparameters live at:

```
config/pipelines/jsonl2vocab.yaml
```

You may override this path via the `-c/--config` flag.

---

## Input Format

- Directory containing one or more JSONL files.  
- Each line must be valid JSON conforming to the `PretrainRecord` schema:
  - `"raw"`: **Required** - the raw text string for vocabulary induction  
  - `"type"`: Optional - content type classification ("natural", "code", "mixed")  
  - `"id"`: Optional - unique record identifier  
  - `"meta"`: Optional - metadata about the segment  

Example record:

```json
{
  "raw": "X-SPANFORMER\nSPAN-AWARE ENCODER\n5.4 Qualitative Span Interpretability…",
  "type": "mixed",
  "id": { "id": "a0409606-f532-4dd2-b02e-2a0bae5bfeee" },
  "meta": { 
    "status": "keep", 
    "doc_language": "en",
    "extracted_by": "pdf2seg",
    "confidence": 0.78,
    "source_file": "XSpanformer_paper.pdf"
  }
}
```

The pipeline validates each record against the `PretrainRecord` schema and gracefully skips invalid records with warnings.

---

## Pipeline Stages & Outputs

All per-stage files are written under the specified output directory (`--out`):

1. **`full_freq/`**  
   - `full_freq.json`: list of `(substring, count)` sorted by descending frequency.  

2. **`candidates/`**  
   - `candidates.txt`: the top `M` multi-char substrings plus all single codepoints.

3. **`pruning/`**  
   - `final_probs.json`: mapping `{ piece: probability }` after EM & pruning.

4. **Root of `--out`**  
   - `vocab.jsonl`: final vocabulary using `VocabPiece` schema, one JSON object per line:
     ```json
     {"piece": "the", "prob": 0.01234}
     {"piece": "X-SPANFORMER", "prob": 0.00056}
     {"piece": "SPAN-AWARE", "prob": 0.00032}
     …
     ```
   - `vocab_stats.json`: comprehensive statistics using `VocabStats` schema:
     ```json
     {
       "total_pieces": 15000,
       "baseline_ppl": 12.45,
       "final_ppl": 10.23,
       "oov_rate": 0.0015,
       "em_iterations": 5,
       "pruned_pieces": 2847
     }
     ```

---

## Mathematical Foundation

This pipeline implements the hybrid Unigram-LM vocabulary induction algorithm described in Section 3.1 of the X-Spanformer paper.

### Key Formulas

**Candidate Set Formation:**
- U₀ = {top M substrings} ∪ {all single codepoints}

**EM Algorithm:**
- **E-step:** Compute best segmentation via Viterbi: `seg*(x) = argmax_seg ∏_{v∈seg} p(v)`
- **M-step:** Update probabilities: `p^(t+1)(u) = freq(u) / total_freq`

**Adaptive Pruning Criteria:**
- **Perplexity:** `PPL = exp(L'/N_p')` where L' is negative log-likelihood, N_p' is total pieces
- **OOV Rate:** `OOV = N_uncov' / N_t` where N_uncov' is uncovered positions, N_t is total codepoints
- **Pruning Condition:** Accept removal if `PPL' - PPL^(t) < τ_ppl` AND `OOV' ≤ δ_oov`

### Coverage Tracking

The algorithm explicitly tracks which codepoint positions are covered by the segmentation, ensuring proper OOV calculation based on uncovered positions rather than unknown vocabulary pieces.

---

## CLI Usage

```bash
uv run -m x_spanformer.pipelines.jsonl2vocab \
  -i <INPUT_DIR> \
  -o <OUTPUT_DIR> \
  [-c <CONFIG_PATH>]
```

### Arguments

- `-i, --in <INPUT_DIR>`  
  Root directory to search for `*.jsonl`.  
- `-o, --out <OUTPUT_DIR>`  
  Output directory for stage subfolders and `vocab.jsonl`.  
- `-c, --config <CONFIG_PATH>` *(optional)*  
  Path to YAML hyperparams file (default: `config/pipelines/jsonl2vocab.yaml`).

---

## Example

Assuming your JSONL segments live under `data/vocab/in` and you want outputs under `data/vocab/out`:

```bash
uv run -m x_spanformer.pipelines.jsonl2vocab \
  -i data/vocab/in \
  -o data/vocab/out
```

After completion:

```
data/vocab/out/
├── full_freq/
│   └── full_freq.json
├── candidates/
│   └── candidates.txt
├── pruning/
│   └── final_probs.json
└── vocab.jsonl
```

---

## Hyperparameter Reference

| Parameter          | Description                                                       | Default            |
|--------------------|-------------------------------------------------------------------|--------------------|
| `L_max`            | Maximum substring length to consider                              | 8                  |
| `M_candidates`     | Number of top-frequency substrings to keep before adding codepoints | 50000             |
| `T_max_iters`      | Number of EM iterations                                           | 5                  |
| `min_piece_prob`   | Pruning threshold ε: remove pieces with probability below this     | 1e-6               |
| `delta_perplexity` | Maximum allowed increase in piece-level PPL when pruning          | 0.01               |
| `delta_oov`        | Maximum allowed OOV rate (fraction of uncovered codepoints)       | 0.001              |

Adjust these in your YAML file to control vocabulary size and pruning strictness.
