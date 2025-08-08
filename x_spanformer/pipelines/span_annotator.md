# Span Annotator Pipeline

## Overview

The Unified Span Annotator Pipeline is a production-ready implementation that combines three-turn hierarchical annotation with robust async session management and comprehensive error handling. It processes text sequences to extract linguistically meaningful spans using X-bar theory principles.

## Key Features

- **Three-turn conversation strategy**: Progressive analysis from word-level → phrase-level → clause-level
- **Async batch processing**: Sequential processing with robust error handling
- **Resume capability**: Automatically resumes from previous progress
- **Comprehensive validation**: Error handling and span validation
- **Real-time telemetry**: Progress tracking and detailed statistics
- **Multiple output formats**: Working files and consolidated results

## Architecture

### Core Components

1. **SpanAnnotatorPipeline**: Main pipeline orchestrator
2. **SpanAnnotatorSession**: Async session management
3. **XBarAnnotator**: X-bar theory-based span extraction
4. **Output Management**: Working files, consolidation, and metadata

### Processing Flow

```
Input (corpus.jsonl) → Load Sequences → Filter by Range → 
Process in Batches → Annotate with XBar → Save Working Files → 
Consolidate Results → Generate Metadata → Output
```

## Usage

### Command Line Interface

```bash
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --range 1-100 \
    --model llama3.2:3b \
    --temperature 0.2 \
    --timeout 180.0 \
    --verbose
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--corpus` | Path | Required | Path to corpus.jsonl file |
| `--output` | Path | Required | Output directory for annotations |
| `--range` | str | Optional | Range specification (e.g., '1-100', '5,10,15') |
| `--model` | str | `llama3.2:3b` | LLM model name |
| `--temperature` | float | `0.2` | Model temperature |
| `--timeout` | float | `180.0` | Conversation timeout (seconds) |
| `--verbose` | flag | False | Enable verbose logging |

### Programmatic Usage

```python
from x_spanformer.pipelines.span_annotator import SpanAnnotatorPipeline
from pathlib import Path

# Initialize pipeline
pipeline = SpanAnnotatorPipeline(
    model_name="llama3.2:3b",
    temperature=0.2,
    conversation_timeout=180.0,
    max_retries=3
)

# Process sequences
stats = await pipeline.process_sequences(
    corpus_file=Path("data/vocab/corpus.jsonl"),
    output_dir=Path("data/annotations"),
    range_spec="1-100",
    resume=True
)
```

## Input Format

### Corpus File (corpus.jsonl)

Each line should contain a JSON object with a PretrainRecord structure:

```json
{
    "raw": "The quick brown fox jumps over the lazy dog.",
    "type": "natural",
    "meta": {
        "sequence_number": 1,
        "repo_name": "example-repo",
        "file_path": "example.txt"
    }
}
```

### Required Fields

- `raw`: The text content to annotate
- `type`: Domain type (`natural`, `code`, `mixed`)
- `meta.sequence_number`: Unique sequence identifier

## Output Structure

### Directory Layout

```
output_directory/
├── working/                    # Individual sequence results
│   ├── sequence-00000001.json
│   ├── sequence-00000002.json
│   └── ...
├── consolidated/               # Final consolidated results
│   └── annotations.jsonl
├── metadata.json              # Pipeline statistics and metadata
└── annotation_pipeline.log   # Detailed processing logs
```

### Working Files

Individual sequence results in `working/` directory:

```json
{
    "sequence_id": 1,
    "raw_text": "The quick brown fox jumps over the lazy dog.",
    "domain_type": "natural",
    "timestamp": "2025-08-07T12:00:00.000Z",
    "status": "completed",
    "span_annotations": [
        {
            "start_pos": 0,
            "end_pos": 3,
            "xbar_class": "D",
            "confidence": 0.95,
            "text": "The"
        }
    ],
    "total_spans": 15,
    "agent_metadata": {}
}
```

### Consolidated Output

Final results in `consolidated/annotations.jsonl`:

```json
{
    "sequence_id": 1,
    "raw": "The quick brown fox jumps over the lazy dog.",
    "domain_type": "natural",
    "span_annotations": [...],
    "total_spans": 15,
    "metadata": {
        "annotation_strategy": "three_turn_unified",
        "model": "llama3.2:3b",
        "timestamp": "2025-08-07T12:00:00.000Z"
    }
}
```

### Metadata File

Pipeline statistics and configuration:

```json
{
    "pipeline": "unified_span_annotator",
    "model": "llama3.2:3b",
    "last_updated": "2025-08-07T12:00:00.000Z",
    "processing_stats": {
        "total_sequences": 100,
        "successful_annotations": 95,
        "failed_annotations": 5,
        "total_spans": 1520,
        "success_rate": 0.95
    },
    "session_stats": {
        "total_sequences_processed": 95,
        "avg_spans_per_sequence": 16.0,
        "total_processing_time": 3600.0
    }
}
```

## X-bar Theory Integration

### Linguistic Foundation

The pipeline uses X-bar theory for syntactic analysis, providing hierarchical span classification:

- **Word Level**: Basic lexical categories (N, V, Adj, Adv, D, P, etc.)
- **Phrase Level**: Intermediate projections (N', V', etc.)
- **Clause Level**: Maximal projections (NP, VP, CP, etc.)

### Domain-Specific Classification

#### Natural Language Domain
- Word: noun, verb, adjective, adverb, determiner, preposition
- Phrase: noun_phrase, verb_phrase, adjective_phrase, adverb_phrase
- Clause: simple_clause, complex_clause, relative_clause
- Sentence: declarative, interrogative, imperative, exclamative

#### Code Domain
- Word: keyword, identifier, literal, operator, delimiter
- Phrase: expression, parameter_list, argument_list
- Statement: assignment, function_call, control_flow, declaration

#### Mixed Domain
- Combines natural and code elements
- Special handling for inline code, documentation, comments

## Three-Turn Annotation Strategy

### Turn 1: Word-Level Analysis
- Identifies basic lexical categories
- Establishes foundation for higher-level analysis
- Focus on individual tokens and morphemes

### Turn 2: Phrase-Level Analysis
- Groups words into syntactic phrases
- Identifies intermediate projections
- Builds on word-level classifications

### Turn 3: Clause-Level Analysis
- Assembles phrases into clauses and sentences
- Identifies complex syntactic structures
- Provides final hierarchical organization

## Configuration

### Model Parameters

- **model_name**: LLM model for annotation (default: `llama3.2:3b`)
- **temperature**: Creativity/randomness parameter (default: `0.2`)
- **conversation_timeout**: Max time per sequence (default: `180.0` seconds)
- **max_retries**: Retry attempts for failed sequences (default: `3`)

### Processing Parameters

- **batch_size**: Sequences processed per batch (fixed: `5`)
- **max_spans_per_sequence**: Limit on spans per sequence (default: `64`)
- **resume**: Resume capability (always enabled)

## Error Handling

### Retry Logic
- Automatic retry for failed sequences
- Exponential backoff for temporary failures
- Graceful degradation for persistent issues

### Validation
- Input format validation
- Span overlap detection
- Character position validation
- Confidence score validation

### Logging
- Comprehensive error logging
- Progress tracking
- Performance metrics
- Debug information (with `--verbose`)

## Performance Considerations

### Memory Management
- Sequential processing to manage memory usage
- Batch processing for efficiency
- Working file cleanup options

### Scalability
- Resume capability for large datasets
- Parallel processing within batches
- Configurable timeouts and retries

### Monitoring
- Real-time progress tracking
- Performance statistics
- Error rate monitoring
- Resource usage tracking

## Dependencies

### Core Dependencies
- `asyncio`: Asynchronous processing
- `pathlib`: File system operations
- `json`: Data serialization
- `logging`: Comprehensive logging

### X-Spanformer Dependencies
- `x_spanformer.schema`: Data structures
- `x_spanformer.agents.session`: Session management
- `x_spanformer.agents.ollama_client`: LLM integration
- `x_spanformer.xbar`: X-bar theory implementation

### External Dependencies
- Ollama service running on `localhost:11434`
- Compatible LLM model (e.g., `llama3.2:3b`)

## Examples

### Basic Usage

```bash
# Process all sequences
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations

# Process specific range
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --range 1-100

# Process with custom model
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --model phi4-mini \
    --temperature 0.1
```

### Advanced Configuration

```bash
# High-precision annotation
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --temperature 0.1 \
    --timeout 300.0 \
    --verbose

# Fast processing with reduced timeout
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --timeout 60.0
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama service is running
   - Check model availability
   - Verify network connectivity

2. **Memory Issues**
   - Reduce batch size
   - Enable resume capability
   - Use smaller sequence ranges

3. **Timeout Errors**
   - Increase conversation timeout
   - Check model responsiveness
   - Monitor system resources

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --verbose
```

## Best Practices

### Data Preparation
- Ensure consistent input format
- Validate sequence numbering
- Check text encoding (UTF-8)

### Processing Strategy
- Start with small ranges for testing
- Resume capability is always enabled for reliability
- Monitor progress and error rates

### Output Management
- Regular backup of working files
- Monitor disk space usage
- Implement result validation

### Performance Optimization
- Adjust timeout based on text complexity
- Use appropriate temperature settings
- Monitor and tune batch processing

## Future Enhancements

### Planned Features
- Parallel batch processing
- Advanced error recovery
- Real-time progress visualization
- Enhanced validation rules
- Custom classifier definitions

### Integration Points
- Embedding pipeline integration
- Evaluation framework compatibility
- Custom output format support
- API endpoint development
