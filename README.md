[![Python package](https://github.com/p3nGu1nZz/x-spanformer/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/p3nGu1nZz/x-spanformer/actions/workflows/python-package.yml)

# 🧠 X-Spanformer

**Tokenizer-free, span-aware encoder architecture inspired by X-bar theory**  
X-Spanformer replaces static tokenization with adaptive, corpus-driven vocabulary induction via expectation-maximization and Viterbi approximation. The system learns to segment and encode overlapping spans directly from raw UTF-8 input—code, natural language, or hybrid text—using factorized pointer networks and controller fusion mechanisms for downstream transformer integration.

---

## 🚀 Key Features

- **Tokenizer-Free Architecture** – adaptive vocabulary induction via hybrid Unigram-LM with EM and Viterbi approximation  
- **Factorized Pointer Networks** – independent span boundary prediction for start and end positions  
- **Multi-Domain Compositional Spans** – supports code, prose, mixed content with X-bar hierarchical structure  
- **Controller Fusion Integration** – multiple injection pathways (bias, prefix, gated-FFN) for downstream transformers  
- **ONNX-Native Implementation** – fully differentiable with linear-time convolutional contextual encoding
- **Chunk-Based Storage System** – efficient compressed storage with automatic validation and resume capabilities
- **Comprehensive Integrity Checking** – ensures no missing sequences with gap detection and repair
- **Fast Sequence Introspection** – millisecond loading of individual sequences for analysis and debugging  

---

## 📦 Data Format

Training examples follow this schema:

```json
{
  "input": ["The", " ", "quick", " ", "brown", " ", "fox", "."],
  "type": "natural_language",
  "span_labels": [
    { "span": [0, 0], "label": "determiner", "role": "specifier", "text": "The" },
    { "span": [2, 4], "label": "adjective_phrase", "role": "modifier", "text": "quick brown" },
    { "span": [6, 6], "label": "noun", "role": "subject", "text": "fox" }
  ]
}
```

For full details, see [`/examples`](./examples) and the companion compiler agent [`ox-bar`](https://github.com/.../ox-bar).

---

## 🧪 Data Preprocessing

Our preprocessing pipeline consists of two main stages:

### Stage 1: PDF to JSONL Conversion

To generate semantically coherent pretraining data without tokenizers, we use the [`pdf2seg`](https://pypi.org/project/pdf2seg) package:

```bash
pip install pdf2seg
```

Process scanned or structured PDFs into entropy-minimized text spans:

```bash
# Generate JSONL segments from PDFs
uv run -m x_spanformer.pipelines.pdf2jsonl \
  -i input_pdfs/ \
  -o data/pretraining/out \
  --name pretraining
```

### Stage 2: Vocabulary Induction

Generate a hybrid Unigram-LM vocabulary from the JSONL segments using the **Adaptive Unigram-LM Vocabulary Induction** algorithm (Section 3.1):

```bash
# Induce vocabulary from JSONL segments
uv run -m x_spanformer.pipelines.jsonl2vocab \
  -i data/pretraining/out \
  -o data/vocab/out
```

This implements the mathematical formulation from Section 3.1 of our paper, featuring:

- **EM with Viterbi approximation** for optimal segmentation with adaptive pruning based on perplexity and OOV thresholds
- **Whitespace-aware tokenization** ensuring strict separation between whitespace and content tokens
- **Frequency-based candidate filtering** retaining top-M substrings up to maximum length L_max
- **Statistical validation** with baseline/final perplexity comparison and coverage metrics
- **Schema-validated vocabulary pieces** using `VocabPiece` and `VocabStats` models
- **Multi-stage artifact generation** for transparency and debugging
- **Consolidated corpus output** (`corpus.jsonl`) ready for downstream vocab2embedding processing

### Stage 3: Seed Embeddings & Span Candidate Generation

Transform vocabulary into contextualized embeddings and span candidates using Section 3.2 algorithms:

```bash
# Generate embeddings from vocabulary and text sequences
uv run -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --output data/embedding \
  --config config/pipelines/vocab2embedding.yaml

# Parallel processing with multiple workers for high throughput
uv run -m x_spanformer.pipelines.vocab2embedding \
  --vocab data/vocab/out/vocab.jsonl \
  --input data/vocab/out/corpus.jsonl \
  --output data/embedding \
  --workers 4 \
  --config config/pipelines/vocab2embedding.yaml
```

This implements the unified algorithm from Section 3.2, featuring:

- **Forward-backward soft probability computation** P[t,i] = (α_t × p(u_i) × β_{t+|u_i|}) / α_{T+1} for probabilistic piece assignment
- **Vocabulary-aware Xavier initialization** with probability-adjusted embedding variance for rare pieces
- **Multi-scale dilated convolutional encoding** for contextual representation (kernels [3,5,7], dilations [1,2,4])
- **Factorized span candidate generation** with vocabulary-informed filtering based on alignment and compositional potential

**Output Structure (Chunk-Based):**
```
data/embedding/
├── chunks/                       # Compressed chunk storage
│   ├── embeddings_000001.npz    # Sequences 1-100
│   ├── embeddings_000002.npz    # Sequences 101-200
│   └── embeddings_000052.npz    # Final chunk (partial)
├── metadata.json                 # Global metadata and chunk information
└── embedding.log                 # Processing log with stage-by-stage validation
```

**Key Features:**
- **Automatic Resume**: Validates existing chunks and continues from where processing left off
- **Final Integrity Verification**: Ensures all sequences are processed correctly with comprehensive gap detection
- **Efficient Analysis Tools**: Sequence introspector with fast single-sequence loading from chunks
- **Performance Optimization**: Optional components can be disabled for storage efficiency

### Introspection and Analysis

Analyze processed embeddings with the integrated sequence introspector:

```bash
# Basic sequence analysis
uv run -m x_spanformer.embedding.sequence_introspector \
  --id 1 --output data/embedding

# Detailed statistical analysis with span coverage
uv run -m x_spanformer.embedding.sequence_introspector \
  --id 5 --output data/embedding --analyze

# Check total processed sequences
uv run -m x_spanformer.embedding.sequence_introspector \
  --id 1 --output data/embedding --list-total

# Verbose output (complete arrays)
uv run -m x_spanformer.embedding.sequence_introspector \
  --id 10 --output data/embedding --analyze --verbose
```

The introspector efficiently loads individual sequences from chunk files without decompressing entire chunks, providing:

- **Fast Single-Sequence Loading**: Loads specific sequences from compressed chunks in milliseconds
- **Comprehensive Analysis**: Embedding quality metrics, span coverage statistics, and array shape validation
- **Chunk Storage Information**: Storage efficiency, compression ratios, and chunk contribution estimates
- **Statistical Insights**: Mean/std analysis, sparsity detection, and span length distribution

### Pipeline Integration

The pipeline outputs both `vocab.jsonl` (final vocabulary with probabilities) and `vocab_stats.json` (comprehensive training statistics), enabling detailed analysis of the vocabulary induction process.

All pipelines utilize shared utilities from `x_spanformer.pipelines.shared` for consistent text processing and schema validation, eliminating code duplication and ensuring data format consistency across the preprocessing workflow.

Use the output as either raw training strings (for unsupervised Phase I) or compile with `oxbar` to produce labeled span records.

This enables X-Spanformer to bootstrap span boundaries from real-world documents with high structural signal, without relying on brittle tokenization.

### Stage 4: Span Annotation for Boundary Prediction Training

Generate X-bar span annotations for supervised training of the factorized pointer network boundary predictor (Section 3.3):

```bash
# Annotate sequences for span boundary training
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100 \
  --config config/pipelines/span_annotator.yaml

# Parallel processing for large-scale annotation
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-1000 &
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1001-2000 &
```

This implements **agentic X-bar span annotation** for factorized pointer network training, featuring:

- **Independent Boundary Prediction**: Generates training targets for start/end position classification using factorized linear heads
- **X-bar Hierarchical Structure**: Domain-specific classifier extraction based on linguistic phrase structure theory
- **Position-wise Binary Classification**: Creates sigmoid-normalized boundary probabilities for BCE loss training
- **Multi-label Span Support**: Handles overlapping spans at different hierarchical levels (word → phrase → clause)

**Output Structure (Individual Working Files):**
```
data/annotations/
├── working/                      # Individual sequence annotations
│   ├── corpus-seq-00000001.json # Working file for sequence 1
│   └── corpus-seq-00000100.json # Working file for sequence 100
├── consolidated/                 # Final training data
│   └── annotations.jsonl        # All successful annotations
├── metadata.json                 # Global progress and statistics
└── span_annotator.log           # Processing log
```

**Key Features:**
- **Bidirectional Context**: Built on X-Spanformer's position-wise embedding architecture where each H[t] contains bidirectional contextual information
- **Boundary Detection Training**: Generates binary targets for start/end position prediction (not span-level embeddings)
- **Multi-label Support**: BCE loss handles overlapping spans at different hierarchical levels

---

## � Testing Framework

X-Spanformer includes comprehensive test coverage organized into focused categories for maintainability and clear separation of concerns.

### Test Organization

- **`tests/pipelines/`** - Data processing pipeline tests
  - `test_pipelines_pdf2jsonl.py` - PDF→JSONL conversion with AI judging
  - `test_pipelines_jsonl2vocab.py` - Vocabulary induction (Section 3.1)
  - `test_pipelines_vocab2embedding.py` - Seed embeddings & span generation (Section 3.2)
  - `test_integration_vocab2embedding.py` - End-to-end integration validation

- **`tests/embedding/`** - Embedding analysis utilities (Section 3.2)
  - `test_pipeline.py` - Complete vocab2embedding pipeline validation
  - `test_sequence_introspector.py` - Chunk-based sequence loading tests
  - `test_embedding_chunk.py` - Chunk management and validation tests

- **`tests/schema/`** - Pydantic schema validation
  - `test_schema.py` - Basic schema validation
  - `test_schema_comprehensive.py` - Comprehensive schema tests  
  - `test_schema_vocab.py` - Vocabulary schema validation

- **`tests/agents/`** - AI agent and content processing
  - `test_agents.py` - Base agent functionality
  - `test_span_annotator.py` - Span annotation pipeline tests
  - `test_e2e_ollama_client.py` - Ollama client integration

- **`tests/config/`** - Configuration system tests
  - `test_span_annotator_config.py` - Configuration loading with logging support

- **`tests/core/`** - Core utilities and configuration
  - `test_config_loader.py` - Configuration loading
  - `test_error_handling.py` - Error handling validation
  - `test_rich_utils.py` - Console output utilities
  - `test_vocab_*.py` - Vocabulary processing utilities

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories  
python -m pytest tests/embedding/      # Embedding tests (Section 3.2)
python -m pytest tests/pipelines/     # Pipeline tests (Sections 3.1, 3.2)
python -m pytest tests/schema/        # Schema validation tests

# Run with verbose output and coverage
python -m pytest tests/ -v --cov=x_spanformer

# Test specific pipeline components
python -m pytest tests/embedding/test_pipeline.py -v
python -m pytest tests/pipelines/test_pipelines_vocab2embedding.py -v
```

### Test Features

- **Mathematical Correctness** - Validates Section 3.1/3.2 algorithms (EM convergence, forward-backward consistency, Xavier initialization)
- **Integration Testing** - End-to-end pipeline validation with synthetic and real data
- **Schema Validation** - Pydantic model testing with edge cases and comprehensive coverage
- **Synthetic Data Generation** - Automated test data creation for consistent, reproducible testing
- **Modular Architecture** - Organized by functionality for easy navigation and maintenance

---

## �🧰 Repository Structure

```
x-spanformer/
├── x_spanformer/
│   ├── pipelines/        # Data processing pipelines
│   │   ├── shared/       # Shared utilities for consistent processing
│   │   │   ├── text_processor.py  # Text splitting and processing utilities
│   │   │   └── jsonl_processor.py # JSONL file handling and corpus management
│   │   ├── pdf2jsonl.py  # PDF → JSONL conversion with AI judging
│   │   ├── jsonl2vocab.py # Hybrid Unigram-LM vocabulary induction
│   │   ├── vocab2embedding.py # Section 3.2: Seed embeddings & span generation
│   │   └── repo2jsonl.py # GitHub repository → JSONL conversion
│   ├── benchmarks/       # Performance benchmarking tools
│   │   ├── benchmark_vocab2embedding.py # Vocab2embedding pipeline benchmark
│   │   ├── benchmark_vocab2embedding.md # Comprehensive usage documentation
│   │   └── README.md     # Benchmarks package overview
│   ├── embedding/        # Embedding analysis & utilities (Section 3.2)
│   │   ├── embedding_utils.py # Loading, analysis, quality metrics
│   │   ├── span_analysis.py   # Span patterns, hierarchy, coverage
│   │   ├── embedding_viz.py   # Visualization tools (optional deps)
│   │   ├── analyze_results.py # CLI analysis workflows
│   │   ├── sequence_introspector.py # Efficient single-sequence chunk loading
│   │   ├── embedding_chunk.py # Chunk management and validation
│   │   └── test_pipeline.py   # Pipeline validation
│   ├── schema/           # Pydantic data models and validation
│   │   ├── pretrain_record.py # Training data schema
│   │   ├── vocab.py      # Vocabulary piece and statistics schemas
│   │   └── ...           # Other schema definitions
│   ├── agents/           # AI agents for content judging and processing
│   ├── controllers/      # Span controller logic
│   └── views/            # Data visualization and inspection
├── config/               # Pipeline configurations
│   └── pipelines/        # YAML configs for data processing
├── data/                 # Training and vocabulary data
│   ├── pretraining/      # Raw segments from PDF processing
│   ├── vocab/            # Vocabulary induction outputs
│   ├── embedding/        # Chunk-based embedding storage
│   └── benchmarks/       # Performance benchmark results (timestamped)
├── docs/                 # Documentation and paper materials
│   ├── vocab_induction.md    # Section 3.1 documentation
│   ├── seed_embeddings.md    # Section 3.2 documentation  
│   ├── pretraining_schema.md # Data format specifications
│   └── paper/            # LaTeX source and compiled paper
├── tests/                # Unit tests and integration tests
│   ├── pipelines/        # Pipeline-specific tests (PDF→JSONL, vocab induction, embeddings)
│   ├── embedding/        # Embedding module tests (Section 3.2 validation)
│   ├── schema/           # Pydantic schema validation tests
│   ├── agents/           # AI agent and content judging tests
│   └── core/             # Core utilities and configuration tests
└── examples/             # Sample data and usage examples
```

---

## 🧪 Pipeline Tools

### Core Pipelines

- **`pdf2jsonl.py`** — Convert PDFs to validated JSONL segments with AI content judging
- **`jsonl2vocab.py`** — Induce hybrid Unigram-LM vocabulary using EM + Viterbi with adaptive pruning
- **`vocab2embedding.py`** — Generate seed embeddings and span candidates (Section 3.2: forward-backward algorithm, vocabulary-aware Xavier initialization, multi-scale contextualization)
- **`repo2jsonl.py`** — Export GitHub repositories to JSONL with shallow cloning and AI judging

### Shared Utilities

- **`shared/text_processor.py`** — Unified corpus loading and text processing across all pipelines for consistency and maintainability

### Validation & Analysis

- **Schema validation** — Pydantic models ensure data consistency across pipelines
- **Rich console output** — Detailed progress tracking and statistics reporting
- **Incremental processing** — Resume interrupted runs and process new data efficiently
- **Dependency management** — All dependencies from `pyproject.toml` are assumed available (matplotlib, seaborn, pandas, gitpython, pdf2seg, etc.)

### Configuration

- **YAML-based configs** — Hyperparameter tuning for vocabulary induction and content judging
- **Modular architecture** — Easy to extend with new processing stages and validation rules  

---

## 🔬 Performance Benchmarking

X-Spanformer includes a comprehensive benchmarking infrastructure for scientific performance analysis and optimization tracking of pipeline components.

### Benchmarks Package

The `x_spanformer.benchmarks` package provides scientific measurement capabilities with:

- **Statistical Analysis**: Multiple runs with mean, standard deviation, and confidence intervals
- **Stage Breakdown**: Detailed timing for pipeline components (forward-backward, seed embedding, convolution, candidate generation)
- **Parallel Processing Analysis**: Compare sequential vs multi-worker performance scaling
- **Historical Tracking**: Timestamped results for optimization progress monitoring
- **Profiling Support**: Optional cProfile integration for bottleneck identification

### Vocab2Embedding Benchmark

Performance analysis for the vocab2embedding pipeline (Section 3.2):

```bash
# Quick performance check (5 runs, 10 sequences)
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --config config/pipelines/vocab2embedding.yaml

# Scientific analysis with profiling (10 runs, 50 sequences)
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --output data/benchmarks \
    --runs 10 --sequences 50 --profile

# Parallel processing benchmark (compare 1 vs 4 workers)
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --runs 5 --sequences 20 --workers 4
```

### Benchmark Output

Results are automatically saved with timestamps for historical tracking:

```
data/benchmarks/
├── vocab2embedding_benchmark_20250723_171732.json
├── vocab2embedding_benchmark_20250723_180145.json
└── vocab2embedding_benchmark_20250723_184521.json
```

**Example Performance Metrics:**
- **Sequential Processing (1 worker)**: 46.7s ± 2.8s for 12 sequences
- **Parallel Processing (4 workers)**: 29.8s ± 1.1s for 12 sequences (36% speedup)
- **Candidates per Sequence**: ~4,500-5,000 (comprehensive coverage)
- **Stage Breakdown**: 40% candidate generation, 40% forward-backward algorithm
- **GPU Memory Scaling**: 4 workers ≈ 4× GPU memory usage per worker
- **Chunk Storage Efficiency**: ~30-60MB per 100-sequence chunk with compression
- **Resume Performance**: Near-instant startup with existing chunk validation
- **Introspection Speed**: <100ms single-sequence loading from chunks
- **Optimization Targets**: Automatically identifies bottlenecks for targeted improvements

### Development Workflow

```bash
# 1. Baseline measurement before optimization
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --runs 3 --sequences 5

# 2. Make code optimizations...

# 3. Validate improvements with detailed analysis
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --runs 10 --sequences 20 --profile

# 4. Test parallel processing scaling
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --runs 5 --sequences 20 --workers 1

python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/out/vocab.jsonl \
    --input data/pretraining/out/jsonl/dataset.jsonl \
    --runs 5 --sequences 20 --workers 4
```

**Documentation**: See [`x_spanformer/benchmarks/benchmark_vocab2embedding.md`](x_spanformer/benchmarks/benchmark_vocab2embedding.md) for comprehensive usage guide.

---

## 🌱 Embedding Module

The embedding module provides comprehensive utilities for working with **vocab2embedding pipeline** (Section 3.2) outputs, enabling analysis, visualization, and debugging of vocabulary-to-embedding transformations.

### Module Structure

- **`embedding_utils.py`** — Core utilities for loading and analyzing embeddings
- **`span_analysis.py`** — Advanced span pattern analysis with hierarchy detection  
- **`embedding_viz.py`** — Rich visualization tools (matplotlib and seaborn assumed available)
- **`analyze_results.py`** — Command-line analysis workflows
- **`test_pipeline.py`** — Comprehensive pipeline validation

### Quick Start

```python
from x_spanformer.embedding import (
    load_embedding_results,
    analyze_embedding_quality,
    SpanAnalyzer
)

# Load vocab2embedding results
result = load_embedding_results("data/embeddings", sequence_id=1)

# Analyze embedding quality
quality = analyze_embedding_quality(result['contextual_embeddings'])
print(f"Mean norm: {quality['mean_embedding_norm']:.3f}")

# Analyze span coverage patterns  
sequence = result['metadata']['sequence']
candidates = result['metadata']['span_candidates']
analyzer = SpanAnalyzer(sequence, candidates)

coverage = analyzer.compute_coverage_statistics()
print(f"Coverage: {coverage['coverage_density']:.1%}")
```

### Command-Line Analysis

```bash
# Analyze specific sequence
python -m x_spanformer.embedding.analyze_results data/embeddings/ --sequence-id 1

# Batch analysis across sequences
python -m x_spanformer.embedding.analyze_results data/embeddings/ --batch --max-sequences 10

# Export embeddings to numpy
python -m x_spanformer.embedding.analyze_results data/embeddings/ --export contextual
```

### Pipeline Testing

```bash
# Test complete pipeline with synthetic data
python x_spanformer/embedding/test_pipeline.py
```

**Expected Output:**
```
🧪 Testing vocab2embedding pipeline
✅ Pipeline initialized successfully  
✅ Processed sequence: 'the quick brown fox'
  Number of candidates: 112
✅ Embedding quality analysis: Mean norm: 16.816
✅ Span coverage analysis: Coverage density: 100.0%
🎉 All tests passed successfully!
```

### Key Features

- **Quality Assessment** — Embedding norms, variance ratios, similarity analysis
- **Span Pattern Analysis** — Hierarchy detection, coverage gaps, overlap patterns  
- **Visualization Suite** — Heatmaps, PCA plots, span coverage maps (matplotlib/seaborn integration)
- **Chunk-Based Loading** — Efficient single-sequence access from compressed chunk storage
- **Batch Processing** — Aggregate statistics across multiple sequences
- **Export Capabilities** — Numpy format, JSON metadata, comprehensive reporting
- **Fast Introspection** — Millisecond loading times with sequence introspector tool

This module bridges Section 3.2 outputs with downstream X-Spanformer components, providing essential debugging and analysis capabilities for span-aware embedding research.

---

## 🔧 External Tools

### [`pdf2seg`](https://pypi.org/project/pdf2seg)

Segment PDF documents into structured clauses using OCR + spaCy:

```bash
pdf2seg -i paper.pdf -o spans/
```

Ideal for extracting domain-specific clause boundaries from scientific papers, REPL transcripts, or code-heavy PDFs. The output is then processed by our `pdf2jsonl` pipeline for validation and schema conformance.

### [`oxbar`](https://github.com/.../ox-bar)

Generate structured span-labeled records using local LLMs:

```bash
oxbar compile input.txt --type mixed --output spans.json
```

Supports retry logic, confidence scoring, and mode switching. Complements our vocabulary induction by providing supervised span labels for training data.

---

## 🧬 Architectural Foundations

- **Linguistics:** X-bar phrase structure theory for hierarchical span organization and compositional boundaries  
- **Statistical Methods:** Expectation-maximization with Viterbi approximation for optimal vocabulary segmentation  
- **Neural Architecture:** Factorized pointer networks with linear-time convolutional contextual encoding  
- **Integration Mechanisms:** Multiple controller fusion pathways (bias injection, prefix tuning, gated-FFN modulation)

---

## 🤝 Contributing

We welcome span explorers, linguistically curious devs, and tokenizer skeptics.

Ways to help:
- Label new examples using `oxbar` or manual annotations  
- Extend the span role taxonomy for underrepresented domains (e.g., REPLs, math, RST)  
- Build new controller fusion heads or injection pathways  
- Analyze span induction across language families, treebanks, or doc formats  
- Visualize structural routing dynamics in longer sequences

Start with [`CONTRIBUTING.md`](./CONTRIBUTING.md) to onboard.

---

## 📄 Citation & License

This research and code are licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

```
Copyright (c) 2025  
TAU SYSTEMS by NAXZYU CORP.
```

### 📚 Zenodo Preprint  

**[https://zenodo.org/records/15750962](https://zenodo.org/records/15750962)**

```bibtex
@misc{rawson2025xspanformer,
  title        = {X-Spanformer: Tokenizer-Free Span Induction with Structural Fusion},
  author       = {Rawson, Kara and Chrzanowski, Aimee},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15750962},
  url          = {https://zenodo.org/records/15750962}
}
```
