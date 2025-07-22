# Vocab2Embedding Pipeline Documentation

## Overview

The `vocab2embedding` pipeline implements **Section 3.2: Seed Embeddings and Candidate Set Generation** from the X-Spanformer paper. This pipeline transforms vocabulary files from the Section 3.1 pipeline into contextualized embeddings and span candidates using a tokenizer-free, span-aware approach.

### Key Innovations

- **Soft Probability Computation**: Forward-backward algorithm for probabilistic piece assignment
- **Vocabulary-Aware Initialization**: Frequency-scaled Xavier initialization for embedding matrices  
- **Multi-Scale Contextualization**: Dilated convolutional encoder with configurable receptive fields
- **Dynamic Span Width**: Corpus-adaptive maximum span width computation
- **Vocabulary-Informed Filtering**: Three-tier span candidate filtering system

## Architecture

### 1. Soft Probability Computation (Section 3.2.1)

The `UnigramLM` class implements the forward-backward algorithm to compute position-wise soft piece probabilities:

```
P[t,i] = Pr(piece u_i starts at position t | sequence, vocabulary)
```

**Mathematical Foundation:**
- **Forward Pass**: `α_{t+1} = Σ_{u_i matches at t} α_t · p(u_i)`
- **Backward Pass**: `β_t = Σ_{u_i matches at t} p(u_i) · β_{t+|u_i|}`
- **Soft Probabilities**: `P[t,i] = (α_t · p(u_i) · β_{t+|u_i|}) / α_{T+1}`

### 2. Seed Embedding Generation (Section 3.2.2)

The `SeedEmbedder` class creates initial embeddings using vocabulary-aware initialization:

```
H^0 = P · W_emb
```

Where:
- `P ∈ R^{T×V}`: Soft probability matrix
- `W_emb ∈ R^{V×d}`: Embedding matrix with frequency-scaled initialization
- `H^0 ∈ R^{T×d}`: Seed embeddings

**Initialization Strategy:**
- **Frequent pieces**: Lower variance (more stable gradients)
- **Single codepoints**: Standard Xavier initialization
- **Multi-character pieces**: Frequency-scaled Gaussian initialization

### 3. Multi-Scale Contextualization (Section 3.2.3)

The `ConvEncoderKernel` applies multi-scale dilated convolutions:

```
H = ConvEncoder(H^0)
```

**Architecture Details:**
- **Default kernels**: `[3, 5, 7]` for hierarchical pattern capture
- **Default dilations**: `[1, 2, 4]` for exponential receptive field growth
- **Total pathways**: `|kernels| × |dilations|` (9 pathways by default)
- **Receptive fields**: Range from 3 positions to 25 positions

### 4. Dynamic Span Width Computation

**Formula:**
```
w_max = max(longest_word_length, max_sequence_length // 2)
```

**Implementation:**
- Analyze corpus for longest whitespace-separated word
- Hard limit at `max_sequence_length // 2` (default: 256 for 512-length sequences)
- Ensures linguistic coverage while maintaining computational efficiency

### 5. Span Candidate Generation (Section 3.2.4)

The `SpanCandidateGenerator` applies three filtering criteria:

1. **Vocabulary Alignment**: `span matches high-probability piece (≥ τ_vocab)`
2. **Compositional Potential**: `segmentation probability ≥ τ_comp`
3. **Whitespace Coherence**: Complete words or phrase boundaries

**Complexity**: `O(T · w_max)` instead of quadratic `O(T²)`

## Configuration

### Default Configuration (from `config/pipelines/vocab2embedding.yaml`)

```yaml
# Neural architecture parameters
architecture:
  embed_dim: 512                    # Embedding dimensions
  conv_kernels: [3, 5, 7]          # Multi-scale kernel sizes
  conv_dilations: [1, 2, 4]        # Dilation rates
  dropout_rate: 0.1                 # Regularization

# Span generation parameters
span_generation:
  tau_vocab: 1e-4                   # Vocabulary alignment threshold
  tau_comp: 1e-6                    # Compositional potential threshold

# Processing configuration
processing:
  device: "cuda"                    # Computation device
  device_id: 0                      # GPU device ID
  batch_size: 64                    # Batch processing size
  max_sequence_length: 512          # Maximum input length

# Numerical stability
numerical:
  epsilon: 1e-12                    # Numerical stability constant
  max_piece_length: 8               # Maximum vocabulary piece length

# Output configuration
output:
  save_intermediate: true           # Save intermediate representations
  save_numpy_arrays: true          # Export .npy files
  save_json_metadata: true         # Include JSON metadata
  add_analysis: false               # Detailed analysis
```

## Input/Output

### Input Format

**Vocabulary File** (`vocab.jsonl` from Section 3.1):
```json
{"piece": "the", "probability": 0.0234}
{"piece": "and", "probability": 0.0198}
{"piece": " ", "probability": 0.1567}
```

**Dataset File** (`dataset.jsonl` with PretrainRecord format):
```json
{"raw": "Hello world example", "type": "text", "meta": {}}
{"raw": "function add(x, y) { return x + y; }", "type": "code", "meta": {}}
```

### Output Structure

```
output_dir/
├── json/                         # JSON metadata files
│   ├── embedding_000001.json
│   └── embedding_000002.json
├── seed/                         # Seed embeddings (.npy)
│   ├── seed_emb_000001.npy
│   └── seed_emb_000002.npy
├── context/                      # Contextual embeddings (.npy)
│   ├── context_emb_000001.npy
│   └── context_emb_000002.npy
├── soft_prob/                    # Soft probabilities (.npy)
│   ├── soft_probs_000001.npy
│   └── soft_probs_000002.npy
└── embedding.log                 # Processing log
```

### JSON Metadata Format

```json
{
  "sequence_id": 1,
  "sequence": "Hello world",
  "sequence_length": 11,
  "num_candidates": 24,
  "span_width": 32,
  "span_candidates": [[0, 5], [6, 11], [0, 11]],
  "timestamp": "2025-01-22T10:30:45Z"
}
```

### Array Shapes

- **Soft Probabilities**: `(sequence_length, vocabulary_size)`
- **Seed Embeddings**: `(sequence_length, embed_dim)`
- **Context Embeddings**: `(sequence_length, embed_dim)`

## Usage

### Basic Usage

```bash
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/vocab.jsonl \
  --input data/pretraining/dataset.jsonl \
  --output data/embeddings \
  --config config/pipelines/vocab2embedding.yaml
```

### Advanced Usage

```bash
# Custom configuration
python -m x_spanformer.pipelines.vocab2embedding \
  -v data/vocab/vocab.jsonl \
  -i data/pretraining/dataset.jsonl \
  -o data/embeddings \
  -c config/custom_vocab2embedding.yaml
```

### Resume Processing

The pipeline automatically detects and resumes from existing outputs:

```bash
# This will skip already processed sequences
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/vocab.jsonl \
  --input data/pretraining/dataset.jsonl \
  --output data/embeddings
```

## Mathematical Details

### Forward-Backward Algorithm

**Forward Probabilities:**
```
α_1 = 1
α_{t+1} = Σ_{u_i: match(x,t,u_i)} α_t · p(u_i)
```

**Backward Probabilities:**
```
β_{T+1} = 1
β_t = Σ_{u_i: match(x,t,u_i)} p(u_i) · β_{t+|u_i|}
```

**Soft Piece Probabilities:**
```
P_{t,i} = (α_t · p(u_i) · β_{t+|u_i|}) / α_{T+1}  if match(x,t,u_i), else 0
```

### Receptive Field Calculations

For kernel size `k` and dilation `d`:
```
Receptive Field = 1 + (k-1) × d
Padding = ((k-1) × d) // 2
```

**Example configurations:**
- Kernel=3, Dilation=1: RF=3, Padding=1
- Kernel=5, Dilation=2: RF=9, Padding=4  
- Kernel=7, Dilation=4: RF=25, Padding=12

### Dynamic w_max Computation

```python
import re

def compute_dynamic_w_max(sequences, max_sequence_length):
    max_word_length = 0
    for sequence in sequences:
        words = re.split(r'\s+', sequence.strip())
        for word in words:
            if word and len(word) > max_word_length:
                max_word_length = len(word)
    
    corpus_based = max_word_length
    sequence_based = max_sequence_length // 2
    return max(corpus_based, sequence_based)
```

## Performance Characteristics

### Computational Complexity

- **Soft Probability Computation**: `O(T × V × L_max)`
- **Seed Embedding**: `O(T × V × d)`
- **Convolution**: `O(T × d² × |K| × |D|)`
- **Span Enumeration**: `O(T × w_max)` (reduced from `O(T²)`)

### Memory Usage

- **Soft Probability Matrix**: `T × V × 4 bytes` (float32)
- **Embeddings**: `T × d × 4 bytes` each for seed and context
- **Kernel Parameters**: Minimal (reused across sequences)

### Typical Performance

With default configuration (embed_dim=512, T=256):
- **Processing Time**: ~0.5-2 seconds per sequence (GPU)
- **Memory per Sequence**: ~10-50 MB depending on vocabulary size
- **GPU Utilization**: High during convolution, moderate during enumeration

## Logging and Monitoring

### Log Levels

The pipeline provides detailed logging across 5 stages:

1. **PIPELINE INITIALIZATION**: Configuration loading and validation
2. **VOCABULARY LOADING**: Vocabulary statistics and component initialization  
3. **SEQUENCE LOADING**: Corpus statistics and preprocessing
4. **DYNAMIC W_MAX COMPUTATION**: Span width calculation details
5. **SEQUENCE PROCESSING**: Per-sequence processing statistics

### Key Metrics Logged

- **Vocabulary**: Size, single codepoint coverage, probability distribution
- **Device**: CUDA availability, selected device, memory usage
- **Sequences**: Count, length statistics, processing rate
- **Dynamic w_max**: Corpus-based vs sequence-based values
- **File Sizes**: Individual array sizes and total storage

### Error Handling

- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals
- **Resume Capability**: Automatic detection of partial processing
- **Validation**: Input format validation and error reporting
- **Resource Management**: Memory cleanup and device management

## Integration

### Upstream Dependencies

- **Section 3.1 Pipeline**: Requires `vocab.jsonl` output
- **Dataset Format**: PretrainRecord schema with `"raw"` field
- **Kernel Package**: Uses `x_spanformer.kernel` for validation and convolution

### Downstream Usage

- **Section 3.3+ Pipelines**: Provides contextualized embeddings and span candidates
- **Analysis Tools**: Compatible with embedding analysis utilities
- **Export Formats**: NumPy arrays for ML frameworks, JSON for metadata

### Schema Compatibility

- **Input**: `PretrainRecord` and `VocabEntry` schemas
- **Output**: `EmbeddingResult` schema with numpy array attachments
- **Configuration**: YAML schema with nested architecture/processing/output sections

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or `max_sequence_length`
2. **Slow Processing**: Check GPU utilization, consider CPU fallback
3. **Large Output Files**: Disable `save_intermediate` or `add_analysis`
4. **Resume Failures**: Check file permissions and disk space

### Configuration Tuning

- **For Large Vocabularies**: Increase GPU memory allocation
- **For Long Sequences**: Adjust `w_max` computation or sequence truncation
- **For Speed**: Reduce conv_kernels/dilations or disable analysis
- **For Quality**: Increase embedding dimensions or add regularization

### Performance Optimization

- **Batch Processing**: Pipeline processes sequences individually but efficiently
- **Memory Management**: Automatic cleanup between sequences
- **Device Selection**: Automatic NVIDIA GPU selection when available
- **Caching**: Vocabulary components cached across sequences

---

This pipeline serves as the critical bridge between statistical vocabulary induction (Section 3.1) and span-aware encoding (Section 3.3+), providing the mathematical foundation for tokenizer-free, adaptive text processing in X-Spanformer.