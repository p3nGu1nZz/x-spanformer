# Vocab2Embedding Benchmark

Performance benchmarking tool for the X-Spanformer vocab2embedding pipeline, providing scientific measurement and optimization analysis capabilities.

## Overview

The `benchmark_vocab2embedding.py` script provides comprehensive performance analysis for the vocab2embedding pipeline implementation from Section 3.2 of the X-Spanformer paper. It includes:

- **Performance Measurement**: Accurate timing with statistical analysis
- **Stage Breakdown**: Detailed timing for pipeline components 
- **Profiling Support**: Optional cProfile integration for bottleneck identification
- **Historical Tracking**: Timestamped results for optimization progress monitoring
- **Scientific Validation**: Statistical significance testing with multiple runs

## Features

### 🔬 **Scientific Benchmarking**
- Multiple benchmark runs with statistical analysis (mean, std dev, min/max)
- Stage-by-stage timing breakdown (forward-backward, seed embedding, convolution, candidate generation)
- Candidate generation metrics (count per sequence, total candidates)
- Confidence intervals and performance stability metrics

### 📊 **Profiling & Analysis**
- Optional cProfile integration (`--profile` flag)
- Bottleneck identification with optimization recommendations
- Memory usage and GPU transfer analysis
- Performance regression detection

### 🗃️ **Result Storage**
- Timestamped JSON output files for historical tracking
- Structured data format compatible with analysis tools
- Git branch and command-line metadata preservation
- Configurable output directory (`--output` flag)

## Usage

### Basic Benchmark

```bash
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/pretraining/in/corpus.jsonl \
    --config config/pipelines/vocab2embedding.yaml
```

### Full Scientific Analysis

```bash
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/pretraining/in/corpus.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --output data/benchmarks \
    --runs 10 \
    --sequences 50 \
    --profile
```

### Quick Performance Check

```bash
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/pretraining/in/corpus.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --runs 3 \
    --sequences 5
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--vocab` | ✅ | - | Path to vocab.jsonl file from Section 3.1 pipeline |
| `--input` | ✅ | - | Path to input corpus.jsonl file with PretrainRecord format |
| `--config` | ✅ | - | Path to vocab2embedding configuration YAML file |
| `--output` `-o` | ❌ | `data/benchmarks` | Output directory for benchmark results |
| `--runs` | ❌ | `5` | Number of benchmark runs for statistical analysis |
| `--sequences` | ❌ | `10` | Number of sequences to process per run |
| `--profile` | ❌ | `false` | Enable cProfile for detailed performance analysis |

## Output Format

### File Naming Convention

Results are saved with timestamps for historical tracking:

```
data/benchmarks/vocab2embedding_benchmark_YYYYMMDD_HHMMSS.json
```

Example: `vocab2embedding_benchmark_20250123_143052.json`

### JSON Structure

```json
{
  "current": {
    "method": "Current Implementation",
    "num_runs": 5,
    "num_sequences": 10,
    "times": [17.23, 17.41, 17.18, 17.35, 17.29],
    "mean_time": 17.292,
    "std_time": 0.089,
    "min_time": 17.18,
    "max_time": 17.41,
    "mean_candidates_per_seq": 11309.2,
    "total_candidates": 565460,
    "detailed_timing": {
      "forward_backward": {"mean": 6.917, "std": 0.036, "total": 34.585},
      "seed_embedding": {"mean": 1.729, "std": 0.009, "total": 8.645},
      "conv_encoding": {"mean": 1.729, "std": 0.009, "total": 8.645},
      "candidate_generation": {"mean": 6.917, "std": 0.036, "total": 34.585}
    }
  },
  "benchmark_config": {
    "num_runs": 5,
    "max_sequences": 10,
    "vocab_path": "data/vocab/vocab.jsonl",
    "input_path": "data/pretraining/in/corpus.jsonl",
    "config_path": "config/pipelines/vocab2embedding.yaml",
    "profile_enabled": false
  },
  "benchmark_metadata": {
    "benchmark_name": "vocab2embedding",
    "timestamp": "2025-01-23T14:30:52.123456",
    "git_branch": "dev-oxbar",
    "command_line": "python -m x_spanformer.benchmarks.benchmark_vocab2embedding ..."
  }
}
```

## Performance Analysis

### Interpreting Results

**Key Metrics:**
- `mean_time`: Average processing time per benchmark run
- `std_time`: Standard deviation (lower = more stable performance)
- `mean_candidates_per_seq`: Average span candidates generated per sequence
- `detailed_timing`: Breakdown by pipeline stage

**Performance Indicators:**
- **Good Performance**: `std_time < 0.5s`, consistent candidate generation
- **Optimization Needed**: High variance in timing, excessive candidate counts
- **Memory Issues**: Increasing times across runs, GPU memory warnings

### Stage Analysis

The benchmark provides estimated timing breakdown:

1. **Forward-Backward** (~40%): Unigram LM probability computation
2. **Seed Embedding** (~10%): Initial embedding generation  
3. **Conv Encoding** (~10%): Multi-scale convolutional processing
4. **Candidate Generation** (~40%): Span filtering and validation

### Optimization Targets

The tool automatically identifies bottlenecks:

```
🎯 Optimization Targets (slowest first):
  1. Candidate Generation: 6.917s
  2. Forward Backward: 6.917s  
  3. Conv Encoding: 1.729s
```

## Historical Analysis

### Tracking Performance Over Time

```bash
# Compare recent benchmarks
ls -la data/benchmarks/vocab2embedding_benchmark_*.json

# Example progression
vocab2embedding_benchmark_20250120_100000.json  # 32.5s ± 2.1s (baseline)
vocab2embedding_benchmark_20250121_143000.json  # 25.8s ± 1.8s (initial optimization)
vocab2embedding_benchmark_20250123_143000.json  # 17.3s ± 0.4s (final optimization)
```

### Performance Regression Detection

Monitor these metrics across benchmarks:
- **Mean Time**: Should trend downward or remain stable
- **Standard Deviation**: Should decrease (more stable)
- **Candidate Quality**: Verify counts remain reasonable

## Integration with Development Workflow

### Before Optimization

```bash
# Establish baseline
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/pretraining/in/corpus.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --runs 5 --sequences 20
```

### After Optimization

```bash
# Validate improvements
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/pretraining/in/corpus.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --runs 10 --sequences 50 --profile
```

## Troubleshooting

### Common Issues

**GPU Memory Errors:**
- Reduce `--sequences` parameter
- Check CUDA memory usage before running
- Ensure proper GPU memory cleanup in pipeline

**Inconsistent Results:**
- Increase `--runs` for better statistical significance  
- Check for background processes affecting performance
- Verify deterministic behavior in pipeline components

**File Not Found Errors:**
- Verify all input file paths exist
- Check that vocab.jsonl is from completed Section 3.1 pipeline
- Ensure output directory has write permissions

## Advanced Usage

### Custom Analysis Scripts

The JSON output can be analyzed with custom scripts:

```python
import json
import matplotlib.pyplot as plt

# Load benchmark results
with open('data/benchmarks/vocab2embedding_benchmark_20250123_143052.json', 'r') as f:
    results = json.load(f)

# Plot timing distribution
times = results['current']['times']
plt.hist(times, bins=10, alpha=0.7)
plt.xlabel('Processing Time (seconds)')
plt.ylabel('Frequency')
plt.title('Vocab2Embedding Performance Distribution')
plt.show()
```

### Continuous Integration

```bash
# Add to CI pipeline
python -m x_spanformer.benchmarks.benchmark_vocab2embedding \
    --vocab data/vocab/vocab.jsonl \
    --input data/pretraining/in/corpus.jsonl \
    --config config/pipelines/vocab2embedding.yaml \
    --runs 3 --sequences 5 \
    --output ci_benchmarks

# Performance regression check
python scripts/check_performance_regression.py ci_benchmarks/
```

## Related Documentation

- [X-Spanformer Architecture](../../docs/paper/): Paper specifications
- [Vocab2Embedding Pipeline](../pipelines/): Implementation details  
- [Testing Guide](../../docs/testing_guide.md): Integration testing
- [Performance Optimization](../../docs/): Optimization techniques
