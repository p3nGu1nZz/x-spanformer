# 🧬 Vocab2Embedding Pipeline

The vocab2embedding pipeline implements **Section 3.2** of the X-Spanformer paper: "**Seed Embeddings and Candidate Set Generation**". This pipeline transforms the vocabulary output from Section 3.1 into contextualized embeddings and span candidates ready for downstream processing.

## 🚀 Overview

The pipeline implements four key components described in the paper:

1. **🧮 Soft Probability Computation** (Section 3.2.1): Forward-backward algorithm adapted from HMMs
2. **🌱 Seed Embeddings** (Section 3.2.2): Vocabulary-aware Xavier initialization  
3. **🔄 Contextual Encoding** (Section 3.2.3): Multi-scale dilated convolutions
4. **📍 Span Candidate Generation** (Section 3.2.4): Vocabulary-informed filtering

## Architecture

### UnigramLM Class
Implements the forward-backward algorithm to compute soft piece probabilities P[t,i] for each position t and vocabulary piece i.

**Key Features:**
- Log-space computation for numerical stability
- Efficient piece matching using prefix trees
- Handles variable-length pieces correctly

**Mathematical Implementation:**
```
α_t = probability of generating sequence[0:t]
β_t = probability of generating sequence[t:T]
P[t,i] = (α_t × p(u_i) × β_{t+|u_i|}) / α_T
```

### SeedEmbedder Class
Generates initial embeddings using vocabulary-aware Xavier initialization (Eq. 4).

**Initialization Strategy:**
- Single codepoints: `std = sqrt(2/d)`
- Multi-codepoint pieces: `std = sqrt(2/(d×p(u)))`
- Higher-probability pieces get more stable gradients

**Forward Pass:**
```
H^0 = P × W_emb
```

### ConvEncoder Class
Multi-scale contextualization using dilated convolutions.

**Architecture:**
- 3 parallel convolution branches: kernels [3,5,7], dilations [1,2,4]  
- Concatenation followed by projection back to embedding dimension
- Residual connections and layer normalization
- GELU activation functions

### SpanCandidateGenerator Class
Filters potential spans using three criteria from Algorithm 3:

1. **Vocabulary Alignment**: `p(span) ≥ τ_vocab`
2. **Compositional Potential**: Product of piece probabilities ≥ `τ_comp`
3. **Whitespace Coherence**: Respects linguistic boundaries

## Usage

### Basic Usage
```python
from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline

# Initialize pipeline
pipeline = Vocab2EmbeddingPipeline('config/pipelines/vocab2embedding.yaml')

# Load vocabulary from Section 3.1 output
pipeline.load_vocabulary('data/vocab/vocab.jsonl')

# Process sequences
result = pipeline.process_sequence("the quick brown fox")

# Access results
embeddings = result['contextual_embeddings']  # Shape: (T, d)
candidates = result['span_candidates']        # List of (start, end) tuples
soft_probs = result['soft_probabilities']     # Shape: (T, |V|)
```

### Command Line Usage
```bash
# Using dataset.jsonl from the Section 3.1 pipeline
python -m x_spanformer.pipelines.vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/embedding/in \
    --output data/embeddings/out \
    --config config/pipelines/vocab2embedding.yaml \
    --device cuda \
    --batch-size 32
```

## Input Format Support

The pipeline supports PretrainRecord format from the Section 3.1 pipeline:

### PretrainRecord Format
Direct usage of dataset.jsonl files from the vocabulary induction pipeline:
```json
{"raw": "the quick brown fox", "type": "text", "id": "001", "meta": {"source": "example.txt"}}
{"raw": "jumps over the lazy dog", "type": "text", "id": "002", "meta": {"status": "validated"}}
```

**Features:**
- Extracts text content from the `raw` field of PretrainRecord entries
- Skips sequences marked as `status: "discard"` in metadata
- Handles empty lines and malformed JSON gracefully
- Logs processing statistics and corpus summary

## Configuration

The pipeline uses YAML configuration files with the following parameters:

```yaml
# Embedding parameters
embed_dim: 256           # Embedding dimension d
dropout_rate: 0.1        # Dropout rate

# Span filtering thresholds
tau_vocab: 1.0e-4        # Vocabulary alignment threshold
tau_comp: 1.0e-6         # Compositional potential threshold  
w_max: 64                # Maximum span width

# Multi-scale convolution
conv_kernels: [3, 5, 7]  # Kernel sizes
conv_dilations: [1, 2, 4] # Dilation rates

# Performance
max_sequence_length: 512
batch_size: 32
device: "cuda"
```

## Output Format

The pipeline generates several output files for each processed sequence:

### JSON Metadata (`embedding_XXXXXX.json`)
```json
{
  "sequence_id": 1,
  "sequence": "the quick brown fox",
  "sequence_length": 18,
  "num_candidates": 45,
  "span_candidates": [[0,3], [0,9], [4,9], ...],
  "soft_probabilities_shape": [18, 5000],
  "seed_embeddings_shape": [18, 256],
  "contextual_embeddings_shape": [18, 256]
}
```

### Numpy Arrays
- `soft_probs_XXXXXX.npy`: Soft piece probabilities (T × |V|)
- `seed_emb_XXXXXX.npy`: Seed embeddings (T × d)  
- `context_emb_XXXXXX.npy`: Contextual embeddings (T × d)

## Implementation Details

### Computational Complexity
Following Proposition 3 from the paper:
- **Time**: O(T × |V| × L_max + T × d² + T × w_max²)
- **Space**: O(T × |V| + T × d)

Where:
- T = sequence length
- |V| = vocabulary size
- L_max = maximum piece length
- d = embedding dimension
- w_max = maximum span width

### Memory Optimization
- Sparse tensor support for probability matrices
- Chunked processing for very long sequences  
- GPU memory streaming for batch processing
- Efficient piece matching with prefix trees

### Numerical Stability
- Log-space forward-backward computation
- Gradient clipping in embedding initialization
- Epsilon constants for zero-probability handling

## Testing

Run comprehensive tests with:
```bash
python -m pytest tests/test_pipelines_vocab2embedding.py -v
```

Tests cover:
- Forward-backward algorithm correctness
- Xavier initialization scaling
- Multi-scale convolution functionality
- Span candidate filtering logic
- End-to-end pipeline integration
- Mathematical formulation compliance

## Integration with Other Pipelines

### Input Dependencies
- **vocab.jsonl**: Output from `jsonl2vocab.py` (Section 3.1)
- **dataset.jsonl**: PretrainRecord format files from the vocabulary induction pipeline

### Output Usage  
- Embeddings feed into span boundary prediction modules
- Candidates provide search space for span selection
- Soft probabilities enable differentiable segmentation

## Performance Considerations

### GPU Acceleration
- All matrix operations are GPU-accelerated
- Batch processing reduces memory overhead
- Mixed precision training supported

### Scaling Recommendations
- **Small datasets** (< 1M sequences): Single GPU sufficient
- **Medium datasets** (1-10M sequences): Multi-GPU data parallelism  
- **Large datasets** (> 10M sequences): Distributed processing recommended

### Memory Usage
Typical memory requirements:
- **CPU**: 2-4 GB for vocabulary + 1-2 GB per sequence batch
- **GPU**: 4-8 GB for model + 2-4 GB per sequence batch

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or max sequence length
   - Enable gradient checkpointing
   - Use CPU fallback for very long sequences

2. **Slow Forward-Backward**
   - Check vocabulary size (prune low-probability pieces)
   - Verify piece length distribution
   - Consider approximate algorithms for very large vocabularies

3. **Poor Span Candidates**
   - Adjust threshold parameters (tau_vocab, tau_comp)
   - Verify vocabulary quality from Section 3.1
   - Check whitespace coherence implementation

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor key metrics:
- Forward-backward convergence  
- Embedding variance after initialization
- Candidate set coverage vs. computational cost

## References

- Section 3.2 of X-Spanformer paper
- Forward-backward algorithm: Baum et al. (1970)
- Xavier initialization: Glorot & Bengio (2010)
- Dilated convolutions: Yu & Koltun (2016)
