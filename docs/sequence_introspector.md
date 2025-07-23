# X-Spanformer Sequence Introspector

A command-line tool for deep introspection and analysis of processed sequences from the vocab2embedding pipeline. This tool provides access to all neural network layers and representations generated during the embedding pipeline processing.

## Overview

The Sequence Introspector allows you to examine the mathematical foundations of X-Spanformer at the sequence level, providing access to:

- **H⁰ (Seed Embeddings)**: Initial dense representations from soft probabilities `P·W_emb` 
- **H (Contextual Embeddings)**: Multi-scale convolutionally enhanced representations
- **P (Soft Probabilities)**: Forward-backward probability matrix from the Unigram-LM
- **Span Candidates**: Filtered candidate set with dynamic w_max

## Installation

The introspector is included with the X-Spanformer package and can be run directly:

```bash
python -m x_spanformer.embedding.sequence_introspector
```

## Basic Usage

### Quick Sequence Overview

```bash
python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out
```

This shows basic information about sequence 1, including:
- Sequence length and preview
- Neural representation shapes (seed embeddings, contextual embeddings)
- Number of span candidates
- Vocabulary size

### List Available Sequences

```bash
python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out --list-total
```

Shows the total number of processed sequences before displaying the selected sequence.

## Advanced Analysis

### Statistical Analysis

```bash
python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out --analyze
```

Provides detailed statistical analysis including:
- **Seed Embeddings**: Mean, standard deviation, range, sparsity
- **Contextual Embeddings**: Statistical distribution after convolutions
- **Soft Probabilities**: Row sums, sparsity (if available)
- **Span Candidates**: Length distribution, quartiles

### Verbose Mode - Complete Float Arrays

```bash
python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out -v
```

Shows the complete seed embeddings array with all float values:
- **No truncation**: All positions and dimensions displayed
- **High precision**: 6 decimal places
- **Full matrix**: Complete (T × 512) seed embeddings

### Combined Analysis

```bash
python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out -v --analyze
```

Combines verbose float display with statistical analysis for comprehensive introspection.

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--id ID` | **Required.** Sequence ID to introspect (1-based indexing) |
| `--output DIR` | **Required.** Path to embedding output directory |
| `--analyze` | Include detailed statistical analysis |
| `-v, --verbose` | Show complete seed embedding float arrays |
| `--list-total` | Display total number of processed sequences |

## Output Structure

The introspector expects the following directory structure from the vocab2embedding pipeline:

```
data/embedding/out/
├── json/           # Metadata files (embedding_XXXXXX.json)
├── seed/           # Seed embedding arrays (seed_emb_XXXXXX.npy)
├── context/        # Contextual embedding arrays (context_emb_XXXXXX.npy)
└── soft_prob/      # Soft probability matrices (soft_probs_XXXXXX.npy) [optional]
```

## Mathematical Context

### Seed Embeddings (H⁰)
From Section 3.2 of the X-Spanformer paper:
```
H⁰ = P · W_emb ∈ ℝ^(T×d)
```
Where:
- `P ∈ ℝ^(T×|V|)` is the soft probability matrix from forward-backward algorithm
- `W_emb ∈ ℝ^(|V|×d)` is the vocabulary embedding matrix
- `T` is sequence length, `d=512` is embedding dimension

### Contextual Embeddings (H)
Multi-scale dilated convolutions applied to seed embeddings:
```
H = ConvEncoder(H⁰) ∈ ℝ^(T×d)
```

## Example Output

### Basic Overview
```
================================================================================
X-SPANFORMER SEQUENCE INTROSPECTOR - SEQUENCE 1
================================================================================

== SEQUENCE INFORMATION:
   Sequence Length: 511 characters
   Span Candidates: 14668
   Vocabulary Size: 15452

== NEURAL REPRESENTATIONS:
   Seed Embeddings (H0):     (511, 512)
   Contextual Embeddings (H): (511, 512)
   Soft Probabilities (P):    (511, 15452)

== SEQUENCE PREVIEW:
   'X-SPANFORMER\nSPAN-AwARE ENCODER\n5.4 Qualitative Span Interpretability...'
```

### Verbose Mode (Float Arrays)
```
== SEED EMBEDDINGS (H0) - COMPLETE FLOAT ARRAY:
   Shape: (511, 512)
   Full array values:
[[-0.218088  0.273766 -0.223549  0.108542 -0.010732  0.193029  0.094456]
 [-0.000000  0.000000  0.000000  0.000000 -0.000000  0.000000 -0.000000]
 [-0.000001 -0.000001  0.000000  0.000001 -0.000001 -0.000001 -0.000000]
 ...
 [-0.047897  0.226180  0.022995  0.151834 -0.337907 -0.113898  0.180403]
 [ 0.026741  0.024113 -0.027639  0.009886 -0.008351  0.012390  0.015332]]
```

### Statistical Analysis
```
== STATISTICAL ANALYSIS:
   Seed Embeddings:
     Mean: 0.000026, Std: 0.056704
     Range: [-0.740066, 0.805024]
     Sparsity: 0.00%

   Contextual Embeddings:
     Mean: -0.000257, Std: 1.026805
     Range: [-4.912900, 4.998506]
     Sparsity: 10.09%

   Span Candidates:
     Total: 14668
     Length: Avg 23.3, Range [1, 84]
     Quartiles: 8.0, 18.0, 35.0
```

## Performance Considerations

### Soft Probabilities
When `save_soft_probabilities: false` in the pipeline configuration (for performance), the introspector will show:
```
   Soft Probabilities (P):    Not saved (performance optimization)
```

The seed and contextual embeddings remain available for analysis.

### Large Sequences
For sequences with large embedding matrices, the verbose mode will display all values. Consider redirecting output to a file for very large sequences:

```bash
python -m x_spanformer.embedding.sequence_introspector --id 1 --output data/embedding/out -v > sequence_1_embeddings.txt
```

## Troubleshooting

### Common Issues

**"Required directory not found"**
- Ensure the output directory contains the expected subdirectories (json/, seed/, context/, soft_prob/)
- Check that the vocab2embedding pipeline has been run successfully

**"Sequence ID exceeds available sequences"**
- Use `--list-total` to see how many sequences have been processed
- Sequence IDs are 1-based, not 0-based

**"File not found" errors**
- Some files may not exist if the pipeline was interrupted
- The introspector requires at least seed and contextual embeddings to function

### File Size Considerations

The introspector loads complete numpy arrays into memory:
- **Seed embeddings**: ~1-2MB per sequence
- **Contextual embeddings**: ~1-2MB per sequence  
- **Soft probabilities**: ~30MB per sequence (if enabled)

For sequences near the maximum length (512 characters), expect higher memory usage.

## Integration with Pipeline

The introspector is designed to work seamlessly with the vocab2embedding pipeline output. After running:

```bash
uv run -m x_spanformer.pipelines.vocab2embedding --vocab vocab.jsonl --input corpus.jsonl --output data/embedding/out --config config/pipelines/vocab2embedding.yaml
```

You can immediately introspect any processed sequence using the tools described above.
