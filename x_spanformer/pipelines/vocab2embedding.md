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
  save_intermediate: false         # Save intermediate representations (deprecated)
  chunk_size: 100                  # Sequences per chunk file
  save_seed_embeddings: false      # Include seed embeddings in chunks (performance optimization)
  save_json_metadata: false        # Include JSON metadata (deprecated in favor of chunk metadata)
  add_analysis: false              # Detailed analysis (deprecated)
  save_soft_probabilities: false   # Include soft probabilities in chunks (performance optimization)
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

The pipeline uses a **chunk-based storage system** for efficient processing and resumption:

```
output_dir/
├── chunks/                       # Chunk-based storage (.npz files)
│   ├── embeddings_000001.npz    # Sequences 1-100
│   ├── embeddings_000002.npz    # Sequences 101-200
│   ├── embeddings_000003.npz    # Sequences 201-300
│   └── embeddings_000052.npz    # Final chunk (partial)
├── metadata.json                 # Global metadata and chunk information
└── embedding.log                 # Processing log
```

**Chunk Structure:**
- Each `.npz` file contains up to 100 sequences
- Compressed storage with contextual embeddings always included
- Optional seed embeddings and soft probabilities (based on configuration)
- Automatic chunk validation and integrity checking
- Efficient single-sequence loading for analysis tools

### Chunk Metadata Format (metadata.json)

```json
{
  "chunk_size": 100,
  "total_sequences": 5107,
  "created_timestamp": "2025-07-25T19:09:45Z",
  "last_updated": "2025-07-25T19:12:23Z",
  "chunks": {
    "1": {
      "file_path": "chunks/embeddings_000001.npz",
      "sequence_range": [1, 100],
      "sequence_count": 100,
      "is_complete": true,
      "components": ["contextual_embeddings", "span_candidates"],
      "created_timestamp": "2025-07-25T19:10:15Z",
      "file_size_mb": 56.2
    },
    "52": {
      "file_path": "chunks/embeddings_000052.npz", 
      "sequence_range": [5101, 5107],
      "sequence_count": 7,
      "is_complete": true,
      "components": ["contextual_embeddings", "span_candidates"],
      "created_timestamp": "2025-07-25T19:12:23Z",
      "file_size_mb": 2.0
    }
  }
}
```

### Chunk File Format (.npz)

Each chunk contains compressed numpy arrays for multiple sequences:

```python
# Loading a chunk file
import numpy as np
data = np.load('embeddings_000001.npz', allow_pickle=True)

# Available arrays:
# - sequence_ids: [1, 2, 3, ..., 100] 
# - sequences: ["text1", "text2", ...]
# - contextual_embeddings: array of shape (N, seq_len, embed_dim)
# - span_candidates: array of candidate spans per sequence
# - seed_embeddings: optional, if save_seed_embeddings=true
# - soft_probabilities: optional, if save_soft_probabilities=true
```

### Array Shapes

- **Soft Probabilities**: `(sequence_length, vocabulary_size)` - optional component
- **Seed Embeddings**: `(sequence_length, embed_dim)` - optional component  
- **Context Embeddings**: `(sequence_length, embed_dim)` - always included
- **Span Candidates**: Variable length list of `[start, end]` tuples per sequence

### Processing Stages

The pipeline executes in 6 distinct stages with comprehensive validation:

1. **STAGE 1: PIPELINE INITIALIZATION** - Configuration and device setup
2. **STAGE 2: VOCABULARY LOADING** - Load and validate vocabulary file
3. **STAGE 3: SEQUENCE LOADING** - Load and validate input sequences  
4. **STAGE 4: DYNAMIC W_MAX COMPUTATION** - Compute corpus-adaptive span width
5. **STAGE 4.5: CHUNK VALIDATION AND REPAIR** - Validate existing chunks and resume processing
6. **STAGE 5: SEQUENCE PROCESSING** - Generate embeddings with parallel workers
7. **STAGE 6: FINAL INTEGRITY VERIFICATION** - Verify all sequences processed correctly

### Resume and Validation Features

The pipeline includes robust resume capabilities and integrity checking:

- **Automatic Resume**: Detects existing chunks and skips processed sequences
- **Chunk Validation**: Verifies completeness of existing chunks before processing
- **Pipeline Testing**: Tests chunk repair process before starting new processing  
- **Final Integrity Check**: Counts all sequences across all chunks after completion
- **Gap Detection**: Identifies and reports any missing sequence ranges
- **Exit Codes**: Returns appropriate codes based on processing and integrity results

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

The pipeline automatically detects and resumes from existing chunk-based outputs:

```bash
# This will validate existing chunks and skip already processed sequences
python -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/vocab.jsonl \
  --input data/pretraining/dataset.jsonl \
  --output data/embeddings
```

**Resume Process:**
1. **Load Existing Metadata**: Read chunk information from `metadata.json`
2. **Validate Chunks**: Verify completeness of each existing chunk file
3. **Detect Missing Sequences**: Identify gaps in sequence processing
4. **Pipeline Testing**: Test chunk repair process before starting new work
5. **Continue Processing**: Process only missing sequences with full validation
6. **Final Verification**: Verify all sequences are present after completion

### Introspection and Analysis

Use the sequence introspector to analyze processed embeddings:

```bash
# Basic sequence analysis
python -m x_spanformer.embedding.sequence_introspector \
  --id 1 --output data/embeddings

# Detailed statistical analysis  
python -m x_spanformer.embedding.sequence_introspector \
  --id 5 --output data/embeddings --analyze

# Check total processed sequences
python -m x_spanformer.embedding.sequence_introspector \
  --id 1 --output data/embeddings --list-total

# Verbose output (complete arrays)
python -m x_spanformer.embedding.sequence_introspector \
  --id 10 --output data/embeddings --analyze --verbose
```

The introspector efficiently loads individual sequences from chunk files without decompressing entire chunks.

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

The pipeline provides detailed logging across 6 stages:

1. **PIPELINE INITIALIZATION**: Configuration loading and validation
2. **VOCABULARY LOADING**: Vocabulary statistics and component initialization  
3. **SEQUENCE LOADING**: Corpus statistics and preprocessing
4. **DYNAMIC W_MAX COMPUTATION**: Span width calculation details
5. **CHUNK VALIDATION AND REPAIR**: Existing chunk verification and missing sequence detection
6. **SEQUENCE PROCESSING**: Per-sequence processing statistics with parallel workers
7. **FINAL INTEGRITY VERIFICATION**: Complete dataset validation and gap detection

### Key Metrics Logged

- **Vocabulary**: Size, single codepoint coverage, probability distribution
- **Device**: CUDA availability, selected device, memory usage (simplified device list format)
- **Sequences**: Count, length statistics, processing rate
- **Dynamic w_max**: Corpus-based vs sequence-based values
- **Chunk Validation**: Per-chunk verification status, missing sequence ranges
- **Processing Progress**: Real-time percentage completion, worker status, sequence timing
- **Final Integrity**: Total sequence count verification, chunk completeness validation
- **File Sizes**: Chunk sizes, compression ratios, total storage usage

### Error Handling

- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals with proper cleanup
- **Chunk-Based Resume**: Automatic detection of partial processing with chunk validation
- **Validation**: Input format validation, chunk integrity checking, and error reporting
- **Resource Management**: Memory cleanup, device management, and worker process cleanup
- **Device Fallback**: Intelligent CPU fallback when CUDA unavailable (CI/CD compatibility)
- **Pipeline Testing**: Pre-processing validation to prevent hanging during chunk repair
- **Integrity Verification**: Post-processing validation to ensure no missing sequences
- **Exit Codes**: 0 (success), 1 (general error), 2 (integrity check failure)

## Integration

### Upstream Dependencies

- **Section 3.1 Pipeline**: Requires `vocab.jsonl` output
- **Dataset Format**: PretrainRecord schema with `"raw"` field
- **Kernel Package**: Uses `x_spanformer.kernel` for validation and convolution

### Downstream Usage

- **Section 3.3+ Pipelines**: Provides contextualized embeddings and span candidates through chunk-based loading
- **Analysis Tools**: Sequence introspector with efficient single-sequence loading from chunks
- **Export Formats**: Compressed chunk files (.npz) for efficient storage and loading
- **Metadata Access**: Global metadata.json for chunk information and processing status

### Schema Compatibility

- **Input**: `PretrainRecord` and `VocabEntry` schemas
- **Output**: Chunk-based compressed storage with `ChunkMetadata` schema
- **Configuration**: YAML schema with nested architecture/processing/output sections
- **Metadata**: Global JSON metadata with chunk information and processing status

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size`, `max_sequence_length`, or `workers`
2. **CUDA Unavailable**: Pipeline automatically falls back to CPU with warning message
3. **Slow Processing**: Check GPU utilization, automatic CPU fallback for CI/CD environments
4. **Large Chunk Files**: Adjust `chunk_size` or disable optional components (`save_seed_embeddings`, `save_soft_probabilities`)
5. **Resume Failures**: Check chunk file integrity, file permissions, and disk space
6. **Worker Process Errors**: Reduce `workers` count if experiencing instability
7. **GPU Memory with Multiple Workers**: Each worker uses full GPU memory - reduce workers if OOM
8. **Chunk Validation Hanging**: Use pipeline testing to identify problematic chunks before processing
9. **Integrity Check Failures**: Review chunk completeness and sequence gaps in final verification
10. **Sequence Introspector Slow**: Updated with efficient single-sequence loading from chunks

### Configuration Tuning

- **For Large Vocabularies**: Increase GPU memory allocation or reduce workers
- **For Long Sequences**: Adjust `w_max` computation or sequence truncation
- **For Speed**: Increase `workers` (if memory allows) or reduce conv_kernels/dilations
- **For Quality**: Increase embedding dimensions or add regularization
- **For CI/CD**: Omit `--device` parameter for CPU fallback when CUDA unavailable
- **For High Throughput**: Use 4-8 workers on high-memory GPU systems
- **For Memory Constraints**: Use single worker (workers: 1) to minimize GPU usage
- **For Storage Efficiency**: Disable optional components (`save_seed_embeddings: false`, `save_soft_probabilities: false`)
- **For Debugging**: Enable chunk validation logging and use sequence introspector for analysis
- **For Large Datasets**: Adjust `chunk_size` (default 100) based on memory and storage requirements

### Performance Optimization

- **Sequential Processing**: Use `workers: 1` for debugging or memory-constrained systems
- **Parallel Processing**: Use `workers: 4-8` for high-throughput production systems
- **Memory Management**: Automatic cleanup between sequences in each worker
- **Device Selection**: CUDA when specified, CPU default with intelligent fallback for compatibility
- **Caching**: Vocabulary components cached across sequences per worker
- **Benchmarking**: Use benchmark tool to find optimal worker count for your system

---

This pipeline serves as the critical bridge between statistical vocabulary induction (Section 3.1) and span-aware encoding (Section 3.3+), providing the mathematical foundation for tokenizer-free, adaptive text processing in X-Spanformer.