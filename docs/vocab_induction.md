# 🧬 Vocabulary Induction in X-Spanformer

X-Spanformer uses a **tokenizer-free approach** to vocabulary construction, implementing the **Adaptive Unigram-LM Vocabulary Induction** algorithm described in Section 3.1 of our paper. This process learns optimal span boundaries directly from raw text without relying on fixed tokenizers or language-specific heuristics.

---

## 🚀 Why Tokenizer-Free?

Traditional transformers rely on fixed tokenizers (BPE, SentencePiece, etc.) that:
- **Lock in segmentation decisions** before training begins
- **Struggle with domain shifts** (code vs. prose vs. hybrid content)
- **Miss structural boundaries** that don't align with statistical patterns
- **Can't adapt dynamically** to new linguistic structures

X-Spanformer's vocabulary induction learns span boundaries **during** training, allowing the model to:
- ✅ **Discover meaningful units** across modalities (code, natural language, mixed)
- ✅ **Adapt to structural patterns** specific to each domain
- ✅ **Learn compositional hierarchies** aligned with X-bar theory
- ✅ **Bootstrap from real documents** with high structural signal

---

## 🧭 Algorithm Overview: Adaptive Unigram-LM

The vocabulary induction pipeline follows **Section 3.1** from our paper, implementing a hybrid Unigram Language Model with EM-based learning and adaptive pruning:

### 🔄 **Phase 1: Candidate Generation**
```
𝒰₀ = {top M valid substrings} ∪ {all single codepoints}
```
- Extract all substrings up to length `L_max` from the corpus
- Apply whitespace-aware tokenization with strict separation principle
- Select top `M` candidates by frequency while ensuring complete coverage
- Include all Unicode codepoints for vocabulary completeness

### 🔄 **Phase 2: EM-Based Learning with Viterbi Approximation**
```
E-step: seg*(x) = argmax_seg ∏_{v∈seg} p^(t)(v)    (Viterbi decoding)
M-step: p^(t+1)(u) = Σ_x γ^(t)(u|x) / Σ_x Σ_v γ^(t)(v|x)
```
- Initialize piece probabilities: `p^(0)(u) = freq(u) / Σ_v freq(v)`
- E-step: Find optimal segmentation using Viterbi decoding for each sequence
- M-step: Update probabilities based on usage mass in optimal segmentations
- Iterate until convergence of log-likelihood

### 🔄 **Phase 3: Adaptive Pruning with Quality Constraints**
```
Prune piece u if: ΔPPLₚᵣᵤₙₑ < τ_ppl AND OOVₚᵣᵤₙₑ ≤ δ_oov
```
- Remove low-probability pieces while monitoring impact on model quality
- Maintain perplexity constraint: `ΔPPLₚᵣᵤₙₑ < τ_ppl` (quality preservation)
- Enforce coverage constraint: `OOVₚᵣᵤₙₑ ≤ δ_oov` (completeness preservation)
- Apply frequency-based filtering to balance vocabulary size vs. coverage

---

## 🧠 Mathematical Foundation

### **Whitespace-Aware Tokenization**
The system enforces strict separation between whitespace and content tokens:
```
𝒲 = {ws···ws | ws ∈ 𝒞_w, k ≥ 1}    (pure whitespace sequences)
𝒩 = {sequences containing no characters from 𝒞_w}    (pure non-whitespace)
```
Where `𝒞_w` includes all Unicode whitespace control characters: `{' ', '\t', '\n', '\r', '\x0b', '\x0c'}`

### **Perplexity Calculation**
Corpus perplexity measures segmentation quality:
```
PPL^(t) = exp(L^(t) / N_p^(t))
```
Where:
- `L^(t)` = negative log-likelihood of segmentation at iteration t
- `N_p^(t)` = total number of pieces in optimal segmentation

### **Coverage Tracking**
Out-of-vocabulary rate based on uncovered codepoint positions:
```
OOV = N_uncov / N_t
```
Where:
- `N_uncov` = number of codepoint positions not covered by vocabulary
- `N_t` = total codepoints in corpus

### **Adaptive Pruning Criteria**
A piece `u` is pruned if both conditions hold:
1. **Perplexity constraint**: `PPL' - PPL^(t) < τ_ppl`
2. **Coverage constraint**: `OOV' ≤ δ_oov`

This ensures vocabulary reduction doesn't significantly harm model quality.

---

## 📊 Pipeline Stages & Artifacts

The `jsonl2vocab` pipeline generates comprehensive artifacts for transparency and debugging:

### **Stage 1: Frequency Analysis**
- **Input**: JSONL files with `raw` text fields
- **Output**: `full_freq/full_freq.json` - All substring frequencies
- **Purpose**: Discover statistical patterns in the corpus

### **Stage 2: Candidate Selection**  
- **Input**: Frequency data + hyperparameters
- **Output**: `candidates/candidates.txt` - Initial vocabulary U₀
- **Purpose**: Form complete vocabulary foundation

### **Stage 3: EM Training**
- **Input**: Candidate set + corpus
- **Output**: `pruning/final_probs.json` - Piece probabilities
- **Purpose**: Learn optimal segmentation via EM algorithm

### **Stage 4: Final Vocabulary**
- **Input**: Trained probabilities + pruning criteria  
- **Output**: 
  - `vocab.jsonl` - Final vocabulary pieces with probabilities
  - `vocab_stats.json` - Comprehensive training statistics
- **Purpose**: Production-ready vocabulary with full provenance

---

## 🎯 Key Advantages

### **🌐 Domain Agnostic**
- Works across code, natural language, and hybrid content
- No language-specific tokenization rules needed
- Adapts to domain-specific structural patterns

### **📈 Statistically Grounded**
- EM algorithm guarantees convergence to local optimum
- Perplexity-based pruning maintains model quality
- Coverage constraints prevent catastrophic vocabulary reduction

### **🔍 Transparent & Debuggable**
- Multi-stage artifacts enable detailed analysis
- Statistics tracking shows algorithm behavior
- Schema validation ensures data consistency

### **🧩 X-Bar Compatible**
- Learns span boundaries that can align with syntactic structure
- Supports hierarchical composition during training
- Enables controller-based structural routing

---

## 🛠️ Configuration & Hyperparameters

Key hyperparameters control the vocabulary induction process:

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `L_max` | Maximum substring length | 8 | Longer = more compositional pieces |
| `M_candidates` | Top frequency substrings | 50000 | Higher = larger initial vocabulary |
| `T_max_iters` | EM iterations | 5 | More = better convergence |
| `min_piece_prob` | Pruning threshold | 1e-6 | Lower = keeps more pieces |
| `delta_perplexity` | PPL increase limit | 0.01 | Lower = stricter quality control |
| `delta_oov` | OOV rate limit | 0.001 | Lower = better coverage |

### **Tuning Guidelines**
- **For code-heavy corpora**: Increase `L_max` to capture function names and identifiers
- **For natural language**: Lower `M_candidates` for more focused vocabulary
- **For mixed content**: Balance `delta_perplexity` vs. `delta_oov` for domain coverage

---

## 📈 Output Analysis

### **VocabStats Schema**
The pipeline outputs comprehensive statistics for analysis:

```json
{
  "total_pieces": 15000,          // Final vocabulary size |V|
  "baseline_ppl": 12.45,          // Initial corpus perplexity PPL^(0) 
  "final_ppl": 10.23,             // Final perplexity after training
  "oov_rate": 0.0015,             // Uncovered codepoint positions
  "em_iterations": 5,             // EM iterations performed  
  "pruned_pieces": 2847           // Pieces removed during adaptation
}
```

### **Quality Metrics**
- **Perplexity Improvement**: `baseline_ppl - final_ppl` shows learning effectiveness
- **Coverage**: `1.0 - oov_rate` shows vocabulary completeness  
- **Efficiency**: `pruned_pieces / total_pieces` shows pruning aggressiveness
- **Convergence**: Low `em_iterations` suggests stable learning

---

## 🚀 Integration with X-Spanformer

The vocabulary induction pipeline connects to the broader X-Spanformer architecture:

### **Phase I: Unsupervised Span Discovery**
- Use raw vocabulary pieces as initial span candidates
- Bootstrap span boundaries from statistical patterns
- Learn controller vectors from induced spans

### **Phase II: Supervised Span Learning**  
- Combine induced vocabulary with labeled span data
- Refine boundaries using structural annotations
- Train span-aware attention mechanisms

### **Phase III: Multi-Modal Composition**
- Apply learned spans across code, text, and hybrid content
- Enable cross-modal structural transfer
- Support domain-adaptive vocabulary expansion

---

## 🧪 Example Workflow

```bash
# 1. Prepare JSONL corpus from PDFs
uv run -m x_spanformer.pipelines.pdf2jsonl \
  -i data/pretraining/in \
  -o data/pretraining/out \
  --name academic_papers

# 2. Induce vocabulary from corpus  
uv run -m x_spanformer.pipelines.jsonl2vocab \
  -i data/vocab/in \
  -o data/vocab/out \
  -c config/pipelines/jsonl2vocab.yaml

# 3. Analyze results
cat data/vocab/out/vocab_stats.json
head -20 data/vocab/out/vocab.jsonl

# 4. Use vocabulary for span training (Section 3.2+)
# python -m x_spanformer.pipelines.vocab2embedding --vocab data/vocab/out/vocab.jsonl ...
```

**Vocabulary Induction Output:**
- ✅ **High-quality vocabulary**: Statistical and compositional pieces learned from data
- ✅ **Comprehensive statistics**: Detailed metrics for analysis and tuning
- ✅ **Transparent artifacts**: Multi-stage outputs for debugging and research
- ✅ **Ready for Section 3.2**: Compatible with seed embedding pipeline
```

---

## 🔬 Research Impact

This vocabulary induction approach enables several research directions:

### **Linguistic Analysis**
- Study how statistical patterns align with syntactic structure
- Compare induced boundaries across languages and domains  
- Analyze the relationship between frequency and compositionality

### **Model Architecture**
- Design span-aware attention mechanisms using induced vocabulary
- Develop controller injection strategies for structural routing
- Explore dynamic vocabulary expansion during training

### **Cross-Domain Transfer**
- Test vocabulary generalization across code/text boundaries
- Measure structural signal preservation in hybrid content
- Develop domain-adaptive vocabulary refinement techniques

---

The vocabulary induction pipeline represents a foundational component of X-Spanformer's tokenizer-free architecture, enabling the model to learn meaningful span boundaries directly from data while maintaining mathematical rigor and statistical grounding. By combining EM-based learning with adaptive pruning, it produces high-quality vocabularies that support both unsupervised discovery and supervised structural learning.
