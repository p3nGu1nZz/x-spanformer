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

The `ConvEncoderKernel` applies multi-scale dilated convolutions with automatic device fallback:

```
H = ConvEncoder(H^0)
```

**Architecture Details:**
- **Default kernels**: `[3, 5, 7]` for hierarchical pattern capture
- **Default dilations**: `[1, 2, 4]` for exponential receptive field growth
- **Total pathways**: `|kernels| × |dilations|` (9 pathways by default)
- **Receptive fields**: Range from 3 positions to 25 positions
- **Device Handling**: CUDA when specified, CPU default with intelligent fallback for CI/CD environments

### 4. Dynamic Span Width Computation

**Formula:**
```
w_max = min(longest_word_length, max_sequence_length // 2)
```

**Implementation:**
- Analyze corpus for longest whitespace-separated word
- Use the smaller value for corpus-adaptive span generation while respecting sequence limits
- Ensures span generation is tailored to actual content while maintaining computational efficiency

### 5. Span Candidate Generation (Section 3.2.4)

The `SpanCandidateGenerator` applies three filtering criteria:

1. **Vocabulary Alignment**: `span matches high-probability piece (≥ τ_vocab)`
2. **Compositional Potential**: `segmentation probability ≥ τ_comp`
3. **Whitespace Coherence**: Complete words or phrase boundaries

**Complexity**: `O(T · w_max)` instead of quadratic `O(T²)`

## Parallel Processing

### Multi-Worker Architecture

The pipeline supports parallel processing through multiple worker processes, each running independent instances of the complete vocab2embedding pipeline. This approach provides:

- **True Parallelization**: Each worker processes different sequences simultaneously
- **GPU Memory Scaling**: With N workers, expect N×GPU memory usage (each worker loads the full model)
- **Process Isolation**: Worker crashes don't affect other workers or the main process
- **Ordered Output**: Results maintain sequence order despite parallel processing

### Configuration

```yaml
processing:
  workers: 1    # Sequential processing (default)
  workers: 4    # 4 parallel workers
  workers: 8    # 8 parallel workers (high-memory systems)
```

### Performance Characteristics

**Sequential Processing (workers: 1)**:
- Lower memory usage (~2-4GB GPU)
- Simpler debugging and profiling
- Predictable resource consumption

**Parallel Processing (workers: N)**:
- **GPU Memory**: N × base memory usage
- **Processing Speed**: ~30-60% improvement with 4-8 workers
- **CPU Overhead**: Worker management and result collection
- **I/O Scaling**: Multiple workers writing to disk simultaneously

### Usage Examples

```bash
# Sequential processing (default)
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --output data/embeddings \
  --workers 1

# Parallel processing (4 workers)
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --output data/embeddings \
  --workers 4

# High-throughput processing (8 workers)
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --output data/embeddings \
  --workers 8
```

### Benchmarking Parallel Performance

The benchmark tool supports testing different worker configurations:

```bash
# Benchmark sequential vs parallel processing
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --config config/pipelines/vocab2embedding.yaml \
  --runs 3 --sequences 8 --workers 1

python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --config config/pipelines/vocab2embedding.yaml \
  --runs 3 --sequences 8 --workers 4
```

**Example Performance Results:**
- **Sequential (1 worker)**: 46.7s for 8 sequences (5.84s per sequence)
- **Parallel (4 workers)**: 29.8s for 8 sequences (3.72s per sequence)
- **Speedup**: 36% improvement with 4 workers

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
  device: "cuda"                    # Device selection ("cuda" or omit for CPU default)
  device_id: 0                      # GPU device ID when using CUDA
  workers: 1                        # Number of parallel worker processes
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

# Parallel processing with 4 workers
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/vocab.jsonl \
  --input data/pretraining/dataset.jsonl \
  --output data/embeddings \
  --workers 4

# High-throughput processing with 8 workers
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/vocab.jsonl \
  --input data/pretraining/dataset.jsonl \
  --output data/embeddings \
  --workers 8 \
  --config config/pipelines/vocab2embedding.yaml
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
    return min(corpus_based, sequence_based)  # Use smaller value for corpus adaptation
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

**Sequential Processing (1 worker):**
With default configuration (embed_dim=512, T=256):
- **Processing Time**: ~0.5-2 seconds per sequence (GPU)
- **Memory per Sequence**: ~10-50 MB depending on vocabulary size
- **GPU Memory Total**: ~2-4 GB for full pipeline
- **GPU Utilization**: High during convolution, moderate during enumeration

**Parallel Processing (4 workers):**
- **Processing Time**: ~30-60% faster than sequential
- **Memory per Sequence**: Same as sequential per worker
- **GPU Memory Total**: ~8-16 GB (4× sequential usage)
- **CPU Overhead**: Worker management and result coordination
- **I/O Scaling**: Multiple workers writing simultaneously

**Performance Scaling:**
- **2 workers**: ~15-25% speedup
- **4 workers**: ~30-40% speedup  
- **8 workers**: ~40-60% speedup (diminishing returns due to overhead)

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
- **Device**: CUDA availability, selected device, memory usage (simplified device list format)
- **Sequences**: Count, length statistics, processing rate
- **Dynamic w_max**: Corpus-based vs sequence-based values
- **File Sizes**: Individual array sizes and total storage

### Error Handling

- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals
- **Resume Capability**: Automatic detection of partial processing
- **Validation**: Input format validation and error reporting
- **Resource Management**: Memory cleanup and device management
- **Device Fallback**: Intelligent CPU fallback when CUDA unavailable (CI/CD compatibility)

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

1. **CUDA Out of Memory**: Reduce `batch_size`, `max_sequence_length`, or `workers`
2. **CUDA Unavailable**: Pipeline automatically falls back to CPU with warning message
3. **Slow Processing**: Check GPU utilization, automatic CPU fallback for CI/CD environments
4. **Large Output Files**: Disable `save_intermediate` or `add_analysis`
5. **Resume Failures**: Check file permissions and disk space
6. **Worker Process Errors**: Reduce `workers` count if experiencing instability
7. **GPU Memory with Multiple Workers**: Each worker uses full GPU memory - reduce workers if OOM

### Configuration Tuning

- **For Large Vocabularies**: Increase GPU memory allocation or reduce workers
- **For Long Sequences**: Adjust `w_max` computation or sequence truncation
- **For Speed**: Increase `workers` (if memory allows) or reduce conv_kernels/dilations
- **For Quality**: Increase embedding dimensions or add regularization
- **For CI/CD**: Omit `--device` parameter for CPU fallback when CUDA unavailable
- **For High Throughput**: Use 4-8 workers on high-memory GPU systems
- **For Memory Constraints**: Use single worker (workers: 1) to minimize GPU usage

### Performance Optimization

- **Sequential Processing**: Use `workers: 1` for debugging or memory-constrained systems
- **Parallel Processing**: Use `workers: 4-8` for high-throughput production systems
- **Memory Management**: Automatic cleanup between sequences in each worker
- **Device Selection**: CUDA when specified, CPU default with intelligent fallback for compatibility
- **Caching**: Vocabulary components cached across sequences per worker
- **Benchmarking**: Use benchmark tool to find optimal worker count for your system

---

This pipeline serves as the critical bridge between statistical vocabulary induction (Section 3.1) and span-aware encoding (Section 3.3+), providing the mathematical foundation for tokenizer-free, adaptive text processing in X-Spanformer.