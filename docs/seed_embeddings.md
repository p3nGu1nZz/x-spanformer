# 🌱 Seed Embeddings & Span Generation in X-Spanformer

X-Spanformer's **Section 3.2** introduces a revolutionary approach to learning contextualized embeddings from raw text without fixed tokenization. The **Vocab2Embedding Pipeline** transforms induced vocabulary pieces into span-aware representations that capture both local composition and global structural patterns.

---

## 🚀 Why Vocabulary-Aware Seed Embeddings?

Traditional transformers initialize embeddings uniformly, ignoring the rich statistical patterns discovered during vocabulary induction. X-Spanformer's approach leverages the **probability-informed initialization** to:

- ✅ **Stabilize rare piece gradients** through probability-adjusted Xavier scaling
- ✅ **Preserve compositional hierarchies** learned in vocabulary induction  
- ✅ **Enable span-aware attention** through multi-scale contextual encoding
- ✅ **Bootstrap structural discovery** using vocabulary-informed candidate filtering

This creates a **seamless bridge** between statistical vocabulary induction (Section 3.1) and structural span learning (Sections 3.3+).

---

## 🧭 Algorithm Overview: Unified Embedding & Candidate Generation

The pipeline implements **Algorithm 4** from Section 3.2, combining four sophisticated components:

### 🔄 **Phase 1: Soft Probability Computation**
```
P[t,i] = (α_t × p(u_i) × β_{t+|u_i|}) / α_T
```
- Adapts the **forward-backward algorithm** from HMMs to compute piece probabilities
- Uses **log-space arithmetic** for numerical stability with long sequences
- Handles **variable-length pieces** efficiently through dynamic programming

### 🌱 **Phase 2: Vocabulary-Aware Seed Embeddings**
```
std_i = sqrt(2 / (d × p(u_i)))     // Multi-codepoint pieces
std_i = sqrt(2 / d)                // Single codepoint pieces  
H^0 = P × W_emb
```
- **Probability-adjusted Xavier initialization** gives rare pieces larger initial variance
- **Soft embedding lookup** using piece probability matrix P
- Creates **smooth interpolation** between discrete vocabulary choices

### 🔄 **Phase 3: Multi-Scale Contextualization**
```
H = ConvEncoder(H^0)    // Kernels [3,5,7], Dilations [1,2,4]
```
- **Three parallel convolution branches** capture patterns at different scales
- **Dilated convolutions** provide exponential receptive field growth
- **Residual connections + LayerNorm** ensure gradient flow and stability

### 📍 **Phase 4: Vocabulary-Informed Span Filtering**
```
Accept span s if: VocabAlign(s) ∨ CompPot(s) ∨ WhitespaceCoherent(s)
```
- **Vocabulary alignment**: `p(span) ≥ τ_vocab`
- **Compositional potential**: Product of piece probabilities ≥ `τ_comp`  
- **Whitespace coherence**: Respects linguistic word boundaries

---

## 🧠 Mathematical Foundation

### **Forward-Backward Algorithm Adaptation**

The soft probability computation extends the classical HMM forward-backward algorithm to handle **variable-length vocabulary pieces**:

**Forward Pass:**
```
α_0 = 1
α_{t+|u_i|} += α_t × p(u_i)    ∀i where u_i matches at position t
```

**Backward Pass:**  
```
β_T = 1
β_t += p(u_i) × β_{t+|u_i|}    ∀i where u_i matches at position t
```

**Soft Probabilities:**
```
P[t,i] = α_t × p(u_i) × β_{t+|u_i|} / α_T
```

This formulation ensures that `∑_i P[t,i] = 1` at each position, creating a **proper probability distribution** over vocabulary pieces.

### **Vocabulary-Aware Xavier Initialization**

Standard Xavier initialization assumes uniform importance across all embeddings. X-Spanformer adjusts the variance based on piece probability:

**Standard Xavier:** `std = sqrt(2/d)`
**X-Spanformer:** `std = sqrt(2/(d × p(u)))`

**Intuition:** Rare pieces (low `p(u)`) get **larger initial variance**, allowing them to contribute meaningfully despite infrequent occurrence. High-probability pieces get **smaller variance** for stable gradients.

### **Multi-Scale Dilated Convolutions**

The contextual encoder uses three parallel branches with exponentially growing receptive fields:

| Branch | Kernel | Dilation | Receptive Field | Pattern Type |
|--------|---------|----------|-----------------|--------------|
| 1      | 3       | 1        | 3               | Local dependencies |
| 2      | 5       | 2        | 9               | Medium-range composition |
| 3      | 7       | 4        | 25              | Long-range structure |

**Padding Formula:** `padding = (kernel - 1) × dilation ÷ 2`

This ensures **length preservation** while capturing multi-scale compositional patterns.

---

## 📊 Pipeline Architecture & Data Flow

```
Raw Text Sequence
       ↓
┌─────────────────┐
│ UnigramLM       │ → Soft piece probabilities P ∈ R^{T×|V|}
│ (Forward-Back)  │
└─────────────────┘
       ↓
┌─────────────────┐
│ SeedEmbedder    │ → Seed embeddings H^0 ∈ R^{T×d}
│ (Vocab-aware)   │
└─────────────────┘
       ↓
┌─────────────────┐
│ ConvEncoder     │ → Contextual embeddings H ∈ R^{T×d}
│ (Multi-scale)   │
└─────────────────┘
       ↓
┌─────────────────┐
│ CandidateGen    │ → Span candidates [(start,end), ...]
│ (Vocab-informed)│
└─────────────────┘
```

### **Stage Artifacts**

Each processing stage produces specific outputs for analysis and downstream use:

| Stage | Output | Format | Purpose |
|-------|---------|---------|---------|
| Forward-Backward | `soft_probs_XXXXXX.npy` | `(T, |V|)` | Piece probabilities for segmentation analysis |
| Seed Embedding | `seed_emb_XXXXXX.npy` | `(T, d)` | Initial representations before contextualization |
| Contextual Encoding | `context_emb_XXXXXX.npy` | `(T, d)` | Final embeddings for downstream tasks |
| Candidate Generation | `embedding_XXXXXX.json` | JSON | Span positions and metadata |

---

## 🎯 Key Innovations & Advantages

### **🔬 Statistically Grounded**
- **EM-derived probabilities** from Section 3.1 inform initialization scaling
- **Forward-backward algorithm** provides principled soft segmentation
- **Perplexity-aware filtering** maintains vocabulary quality

### **🧩 Compositionally Aware**
- **Multi-scale convolutions** capture hierarchical span structure
- **Vocabulary-informed candidates** respect statistical boundaries
- **Probability interpolation** enables smooth structural transitions

### **⚡ Computationally Efficient**
- **Log-space arithmetic** prevents numerical underflow
- **Parallel convolution branches** leverage GPU acceleration
- **Sparse candidate filtering** reduces downstream computational cost

### **🔍 Interpretable & Debuggable**
- **Soft probabilities** show piece assignment confidence
- **Multi-stage artifacts** enable detailed pipeline analysis
- **Vocabulary alignment scores** explain candidate selection

---

## 🛠️ Configuration & Hyperparameters

### **Core Parameters**

| Parameter | Description | Default | Impact on Learning |
|-----------|-------------|---------|-------------------|
| `embed_dim` | Embedding dimension d | 256 | Higher = more capacity, slower training |
| `tau_vocab` | Vocabulary alignment threshold | 1e-4 | Lower = more candidates, more noise |
| `tau_comp` | Compositional potential threshold | 1e-6 | Lower = more compositional spans |
| `w_max` | Maximum span width | 64 | Higher = longer spans, quadratic cost |

### **Multi-Scale Convolution**

| Parameter | Description | Default | Pattern Capture |
|-----------|-------------|---------|-----------------|
| `conv_kernels` | Kernel sizes | [3, 5, 7] | Local → medium → long dependencies |
| `conv_dilations` | Dilation rates | [1, 2, 4] | Dense → sparse → very sparse sampling |
| `dropout_rate` | Regularization | 0.1 | Higher = more regularization |

### **Performance Tuning**

| Parameter | Description | Default | Scaling Behavior |
|-----------|-------------|---------|------------------|
| `max_sequence_length` | Input length limit | 512 | Quadratic memory in span generation |
| `batch_size` | Sequences per batch | 32 | Higher = better GPU utilization |
| `device` | Computation device | "cuda" | GPU highly recommended for speed |

---

## 🧪 Usage Examples & Workflows

### **Basic Pipeline Execution**

```bash
# 1. Process vocabulary from Section 3.1
python -m x_spanformer.pipelines.vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/sequences.jsonl \
    --output data/embeddings/ \
    --config config/pipelines/vocab2embedding.yaml \
    --device cuda \
    --batch-size 32
```

### **Python API Usage**

```python
from x_spanformer.pipelines.vocab2embedding import Vocab2EmbeddingPipeline

# Initialize pipeline
pipeline = Vocab2EmbeddingPipeline('config/vocab2embedding.yaml', 'cuda')
pipeline.load_vocabulary('data/vocab/vocab.jsonl')

# Process single sequence
result = pipeline.process_sequence("The quick brown fox jumps over the lazy dog.")

# Access results
soft_probs = result['soft_probabilities']        # Shape: (T, |V|)
seed_embeddings = result['seed_embeddings']      # Shape: (T, d)
context_embeddings = result['contextual_embeddings']  # Shape: (T, d)
span_candidates = result['span_candidates']      # List: [(start, end), ...]

print(f"Generated {len(span_candidates)} candidate spans")
print(f"Embedding dimension: {context_embeddings.shape[1]}")
```

### **Advanced Configuration**

```yaml
# config/pipelines/vocab2embedding.yaml
embed_dim: 512                    # Larger embeddings for complex domains
dropout_rate: 0.15               # More regularization

# Span filtering - adjust for domain
tau_vocab: 5.0e-5               # Stricter vocabulary alignment
tau_comp: 1.0e-5                # Looser compositional threshold
w_max: 128                      # Longer spans for code/technical text

# Multi-scale convolution - capture longer dependencies  
conv_kernels: [3, 5, 7, 9]     # Additional long-range kernel
conv_dilations: [1, 2, 4, 8]   # Higher dilation for technical text

# Performance
max_sequence_length: 1024       # Handle longer documents
batch_size: 16                  # Adjust for GPU memory
device: "cuda"                  # Essential for reasonable speed
```

---

## 📈 Output Analysis & Quality Metrics

### **Embedding Quality Assessment**

The pipeline outputs comprehensive statistics for embedding quality analysis:

```json
{
  "sequence_id": 1,
  "sequence": "The quick brown fox jumps over the lazy dog.",
  "sequence_length": 44,
  "num_candidates": 127,
  "span_candidates": [[0,3], [0,9], [4,9], [10,15], ...],
  "soft_probabilities_shape": [44, 15000],
  "seed_embeddings_shape": [44, 256], 
  "contextual_embeddings_shape": [44, 256]
}
```

### **Key Quality Indicators**

**Probability Distribution Health:**
- `max(soft_probs[t,:])`: Peak probability per position (should be > 0.1)
- `entropy(soft_probs[t,:])`: Segmentation uncertainty (lower = more confident)
- `sum(soft_probs[t,:])`: Normalization check (should be ≈ 1.0)

**Embedding Variance:**
- `std(seed_embeddings)`: Initial embedding spread
- `std(contextual_embeddings)`: Post-contextualization spread  
- `corr(seed, contextual)`: How much contextualization changed representations

**Candidate Coverage:**
- `num_candidates / sequence_length²`: Candidate density
- `avg_span_length`: Average span size (should reflect domain patterns)
- `vocab_alignment_rate`: Fraction passing vocabulary filter

### **Debugging Common Issues**

**❌ Problem: Very few candidates generated**
- **Solution**: Lower `tau_vocab` and `tau_comp` thresholds
- **Check**: Vocabulary coverage of input domain

**❌ Problem: Soft probabilities are too uniform**  
- **Solution**: Verify vocabulary quality from Section 3.1
- **Check**: Forward-backward normalization `α_T`

**❌ Problem: Embeddings have very low variance**
- **Solution**: Check vocabulary-aware initialization scaling  
- **Check**: Piece probability distribution in vocabulary

**❌ Problem: GPU memory errors**
- **Solution**: Reduce `max_sequence_length` or `batch_size`
- **Alternative**: Process on CPU for very long sequences

---

## 🔬 Integration with X-Spanformer Architecture

### **Section 3.1 → 3.2 Data Flow**

```
jsonl2vocab Pipeline Output:
├── vocab.jsonl                    # Piece → probability mapping
├── vocab_stats.json              # Training statistics  
└── pruning/final_probs.json      # Detailed probability data

        ↓ (feeds into)

vocab2embedding Pipeline Input:
├── vocab.jsonl                    # Loaded as vocabulary dictionary
└── sequences.jsonl                # Raw text to process

        ↓ (produces)

vocab2embedding Pipeline Output:  
├── embedding_XXXXXX.json         # Metadata + candidates
├── soft_probs_XXXXXX.npy         # Piece probabilities
├── seed_emb_XXXXXX.npy           # Initial embeddings
└── context_emb_XXXXXX.npy        # Contextualized embeddings
```

### **Section 3.2 → 3.3+ Integration**

The vocab2embedding outputs become inputs for downstream X-Spanformer modules:

**Span Boundary Prediction (Section 3.3):**
- `contextual_embeddings`: Feature vectors for boundary classifiers
- `span_candidates`: Candidate set for boundary refinement

**Controller Vector Learning (Section 3.4):**
- `soft_probabilities`: Differentiable segmentation for controller injection
- `contextual_embeddings`: Base representations for controller conditioning

**Span-Aware Attention (Section 3.5):**
- `span_candidates`: Attention mask construction
- `contextual_embeddings`: Query/key/value computation base

---

## 🧬 Research Applications & Extensions

### **Linguistic Analysis**

**Cross-Domain Vocabulary Transfer:**
```python
# Compare embeddings across code vs. natural language
code_pipeline = Vocab2EmbeddingPipeline('config/code_vocab2emb.yaml')
text_pipeline = Vocab2EmbeddingPipeline('config/text_vocab2emb.yaml')

code_result = code_pipeline.process_sequence("def fibonacci(n):")
text_result = text_pipeline.process_sequence("The fibonacci function")

# Analyze embedding space differences
embedding_similarity = cosine_similarity(code_result['contextual_embeddings'], 
                                       text_result['contextual_embeddings'])
```

**Span Hierarchy Discovery:**
```python
# Analyze candidate span nesting patterns
def analyze_span_hierarchy(candidates, sequence):
    nested_spans = []
    for i, (start1, end1) in enumerate(candidates):
        for j, (start2, end2) in enumerate(candidates):
            if start1 <= start2 and end2 <= end1 and i != j:
                nested_spans.append((i, j))  # j is nested in i
    return nested_spans
```

### **Model Architecture Research**

**Alternative Contextualization Strategies:**
- **Self-Attention**: Replace dilated convolutions with multi-head attention
- **Graph Neural Networks**: Model explicit span dependency graphs  
- **Recurrent Contextualization**: Use LSTM/GRU for sequential processing

**Adaptive Span Selection:**
- **Learned Thresholds**: Train neural networks to predict optimal `tau_vocab`, `tau_comp`
- **Reinforcement Learning**: Optimize candidate selection for downstream task performance
- **Active Learning**: Select most informative spans for human annotation

### **Domain-Specific Optimizations**

**Code Understanding:**
```yaml
# Specialized config for programming languages
embed_dim: 512                    # Larger for complex identifier patterns
w_max: 256                       # Capture long function signatures
tau_vocab: 1.0e-5               # Accept more identifier-like spans
conv_kernels: [3, 5, 7, 11]    # Additional kernel for nested structures
```

**Scientific Text:**
```yaml
# Optimized for mathematical notation and citations  
embed_dim: 384                   # Balanced capacity
w_max: 64                       # Moderate span lengths
tau_comp: 5.0e-6               # Allow complex mathematical expressions
conv_dilations: [1, 3, 9]      # Skip over formulaic patterns
```

---

## 📚 References & Related Work

### **Theoretical Foundations**
- **Forward-Backward Algorithm**: Baum et al. (1970) - "A Maximization Technique in HMMs"
- **Xavier Initialization**: Glorot & Bengio (2010) - "Understanding the difficulty of training deep feedforward neural networks"  
- **Dilated Convolutions**: Yu & Koltun (2016) - "Multi-scale context aggregation by dilated convolutions"

### **Tokenizer-Free Methods**
- **ByT5**: Xue et al. (2022) - "ByT5: Towards a token-free future with pre-trained byte-to-byte models"
- **CANINE**: Clark et al. (2021) - "CANINE: Pre-training an Efficient Tokenization-Free Encoder"
- **CharFormer**: Tay et al. (2022) - "Charformer: Fast character transformers via gradient-based subword tokenization"

### **Span-Based Models**  
- **SpanBERT**: Joshi et al. (2020) - "SpanBERT: Improving pre-training by representing and predicting spans"
- **ELECTRA**: Clark et al. (2020) - "ELECTRA: Pre-training text encoders as discriminators rather than generators"

---

## 🚀 Future Directions

### **Immediate Extensions (Version 2.0)**
- **Batch Processing**: Parallel sequence processing for improved throughput
- **Mixed Precision**: FP16 training for memory efficiency  
- **Distributed Training**: Multi-GPU support for large-scale processing

### **Advanced Research (Version 3.0+)**
- **Neural Vocabulary Induction**: End-to-end differentiable vocabulary learning
- **Cross-Modal Embeddings**: Joint text-code-image span representations
- **Dynamic Span Selection**: Task-adaptive candidate filtering

### **Production Features**
- **Streaming Processing**: Handle infinite text streams
- **Incremental Vocabulary**: Online vocabulary expansion
- **Model Serving**: REST API for real-time embedding generation

---

The **Seed Embeddings & Span Generation** pipeline represents a critical bridge between statistical vocabulary induction and structural span learning in X-Spanformer. By combining principled mathematical foundations with efficient implementation, it enables the discovery of meaningful spans across diverse text modalities while maintaining full interpretability and control over the learning process.

This approach opens new research directions in **tokenizer-free language modeling**, **cross-modal representation learning**, and **linguistically-informed neural architectures** — positioning X-Spanformer at the forefront of next-generation language understanding systems.
