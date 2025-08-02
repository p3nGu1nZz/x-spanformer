# Span Annotator Pipeline

## Overview

The X-Spanformer span annotator pipeline implements **agentic X-bar span annotation** for generating supervised training data for the factorized pointer network span predictor. This pipeline processes raw Unicode sequences from corpus.jsonl through multi-turn conversations with phi4-mini to generate hierarchical X-bar span boundary annotations.

### Key Features

- **Comprehensive Classifier Extraction**: Processes each sequence for ALL applicable XBar classifiers
- **Domain-Aware Processing**: Automatic domain detection (natural/code/mixed) with tailored prompts  
- **Agent Pattern Integration**: Uses session-based agents, ollama_client.py, and dialogue.py patterns
- **Scalable Architecture**: Designed for 50k+ LLM requests with statistical tracking
- **Resumable Processing**: Individual working files enable inspection and continuation
- **Multi-Process Support**: Parallel processing on different sequence ranges
- **Bidirectional Contextual Foundation**: Built on X-Spanformer's position-wise embedding architecture

## Architecture

### Bidirectional Contextual Foundation

The annotation system is built on X-Spanformer's bidirectional contextual embedding architecture:

- **Position Embeddings**: Each H[t] ∈ R^512 represents bidirectional contextual character information
- **Multi-scale Context**: Dilated convolutions capture patterns at multiple scales with full sequence awareness
- **Boundary Detection**: Start and end positions access both preceding and following contextual information
- **No Span Averaging**: Spans detected through boundary prediction, not embedding aggregation

### SpanAnnotatorSession

The core annotation engine follows async conversation patterns from the agents session package:

```python
from x_spanformer.agents.session.span_annotator_session import SpanAnnotatorSession
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.xbar.position_mapper import PositionMapper

# Initialize session with dialogue management
agent = SpanAnnotatorSession(
    model_name="phi4-mini",
    max_concurrent=5,
    max_retries=3,
    conversation_timeout=30.0
)

# Process sequences asynchronously
async for result in agent.annotate_batch(sequences):
    # Handle span annotations with position alignment
    for annotation in result.span_annotations:
        print(f"Span: {annotation.start_pos}-{annotation.end_pos}")
```

### Pipeline Architecture

The main pipeline class orchestrates the entire annotation process:

```python
from x_spanformer.pipelines.span_annotator import SpanAnnotatorPipeline

# Initialize pipeline
pipeline = SpanAnnotatorPipeline("config/pipelines/span_annotator.yaml")

# Process sequence range
results = await pipeline.process_sequence_range(
    corpus_file=Path("data/vocab/corpus.jsonl"),
    output_dir=Path("data/annotations"),
    range_spec="1-100"
)
```

## Position-wise Alignment

### Character-to-Position Mapping

The system uses the `PositionMapper` to convert character-level spans to position-wise indices:

```python
from x_spanformer.xbar.position_mapper import PositionMapper

mapper = PositionMapper(text=original_text)

# Character-level span from LLM agent
char_span = CharacterSpan(
    start_char=4,
    end_char=18,
    xbar_class="NP",
    text="quick brown fox"
)

# Convert to position-wise for training
pos_span = mapper.char_to_position(char_span)
# Result: PositionSpan(start_pos=4, end_pos=19, xbar_class="NP")
```

### Training Target Generation

Position spans are converted to binary boundary targets for training:

```python
# Position span becomes training targets
y_start = torch.zeros(sequence_length)
y_end = torch.zeros(sequence_length)

y_start[pos_span.start_pos] = 1.0      # Start boundary
y_end[pos_span.end_pos - 1] = 1.0      # End boundary (inclusive)

# Loss computation
loss_start = BCE(model.predict_start(H), y_start)
loss_end = BCE(model.predict_end(H), y_end)
```

## XBar Classifier Integration

### Comprehensive Classifier Mapping

The pipeline uses comprehensive classifier definitions from `x_spanformer.agents.xbar_map`:

```python
from x_spanformer.xbar.xbar_map import XBarClassifierMap, DomainType

# Get classifiers for domain
classifiers = XBarClassifierMap.get_classifiers_for_domain(DomainType.NATURAL)

# Build system prompt
system_prompt = XBarClassifierMap.build_system_prompt(DomainType.NATURAL)
```

#### Natural Language Classifiers

- **Word Level**: noun, verb, adjective, adverb, determiner, preposition, pronoun, conjunction, punctuation
- **Phrase Level**: noun_phrase, verb_phrase, adjective_phrase, adverb_phrase, prepositional_phrase
- **Clause Level**: main_clause, subordinate_clause, relative_clause
- **Sentence Level**: simple_sentence, compound_sentence, complex_sentence

#### Code Classifiers

- **Word Level**: keyword, identifier, operator, literal, delimiter, type_name, comment
- **Phrase Level**: expression, function_call, assignment, parameter_list, argument_list
- **Statement Level**: if_statement, loop_statement, function_definition, class_definition, import_statement, return_statement

#### Mixed Domain Classifiers

- **Mixed Content**: inline_code, code_block, natural_instruction, documentation_comment, api_reference, error_message

## Processing Strategy

### Comprehensive Sequence Processing

Each sequence undergoes systematic processing through a single comprehensive conversation:

1. **Domain Detection**: Automatic detection from corpus.jsonl type field
2. **Classifier Map Building**: Build complete XBar classifier mapping for domain
3. **System Prompt Construction**: Create comprehensive prompt with ALL classifier definitions
4. **Single Async Conversation**: One multi-turn conversation covering all applicable classifiers
5. **Response Parsing**: Parse comprehensive JSON response and validate span boundaries
6. **Position Alignment**: Convert character spans to position-wise spans for training
7. **Statistics Tracking**: Track success/failure rates per classifier type

**Efficiency Benefits**:
- **Single Request**: One conversation per sequence instead of 10-30 individual requests
- **Context Preservation**: Model maintains awareness of all classifier types throughout annotation
- **Reduced Latency**: ~95% reduction in round-trip requests (1 vs ~25 per sequence)
- **Better Consistency**: Model can ensure non-overlapping spans across classifier types

### Multi-Process Architecture

The pipeline supports parallel processing through range-based sequence assignment:

```python
def process_sequence_range(corpus_file: Path, output_dir: Path, sequence_range: str):
    """Process a specific range of sequences with comprehensive classifier extraction."""
```

## Input/Output

### Input Format

**Corpus File** (corpus.jsonl from jsonl2vocab pipeline):
```json
{"raw": "The quick brown fox jumps over the lazy dog.", "type": "natural", "id": {"id": "corpus-seq-00000001"}, "meta": {"sequence_number": 1}}
{"raw": "function add(x, y) { return x + y; }", "type": "code", "id": {"id": "corpus-seq-00000002"}, "meta": {"sequence_number": 2}}
{"raw": "To define a function `add(x, y)` that returns the sum:", "type": "mixed", "id": {"id": "corpus-seq-00000003"}, "meta": {"sequence_number": 3}}
```

### Output Structure

The pipeline uses individual working files for easy inspection and parallel processing:

```
output_dir/
├── working/                        # Individual sequence annotation files
│   ├── corpus-seq-00000001.json   # Working annotations for sequence 1
│   ├── corpus-seq-00000002.json   # Working annotations for sequence 2
│   └── corpus-seq-00000003.json   # Working annotations for sequence 3
├── consolidated/                   # Final training data
│   └── annotations.jsonl          # All annotations in training format
├── metadata.json                   # Global progress and statistics
└── span_annotator.log             # Processing log
```

### Working File Format

Each working file contains comprehensive annotation results from a single conversation:

```json
{
  "corpus_id": "corpus-seq-00000001",
  "sequence_number": 1,
  "raw_sequence": "The quick brown fox jumps over the lazy dog.",
  "domain_type": "natural",
  "source_meta": {
    "status": "corpus",
    "source": "data/vocab/corpus.jsonl"
  },
  "annotation_session": {
    "started_at": "2025-08-01T10:30:00.000Z",
    "completed_at": "2025-08-01T10:30:05.123Z", 
    "annotation_status": "completed",
    "model": "phi4-mini",
    "error_message": null
  },
  "annotation_result": {
    "span_annotations": [
      {
        "start_pos": 0,
        "end_pos": 3,
        "xbar_class": "determiner",
        "confidence": 0.95,
        "text": "The"
      },
      {
        "start_pos": 4,
        "end_pos": 19,
        "xbar_class": "noun_phrase", 
        "confidence": 0.92,
        "text": "quick brown fox"
      }
    ],
    "total_positions": 43,
    "processing_time": 5.123
  }
}
```

## Configuration

### Pipeline Configuration (span_annotator.yaml)

```yaml
# Model configuration
model:
  name: "phi4-mini"
  temperature: 0.1
  max_tokens: 1024
  timeout: 30

# Processing configuration
processing:
  max_retries: 3
  conversation_timeout: 30.0
  batch_size: 64

# Output configuration
output:
  save_working_files: true
  save_failed_requests: true
  consolidate_on_completion: true
  include_metadata: true

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_to_file: true

# X-bar classifier configuration
classifiers:
  natural_labels: ["noun", "verb", "adjective", "adverb", "determiner", "preposition", "pronoun", "conjunction", "punctuation"]
  natural_roles: ["subject", "predicate", "modifier", "complement"]
  code_labels: ["keyword", "identifier", "operator", "literal", "delimiter", "type_name", "comment"]  
  code_roles: ["function", "declaration", "expression", "statement"]
  mixed_roles: ["inline_code", "documentation_context", "instructional_sequence"]
  xbar_roles: ["head", "specifier", "modifier", "complement", "adjunct"]
```

### Agent Configuration (config/agents/span_annotator.yaml)

```yaml
agent_type: span_annotator

model:
  name: phi4-mini
  temperature: 0.1
  max_context_tokens: 131072
  max_turn_tokens: 1024

dialogue:
  max_turns: 8
  memory_limit: 16
  trim_strategy: rolling

span_annotator:
  model_name: phi4-mini
  temperature: 0.1
  max_retries: 3
  conversation_timeout: 30.0
  domain_detection: auto
  comprehensive_annotation: true

format:
  expected_fields: [ text, start_char, end_char, xbar_class, confidence ]
  strict_json: true
  parse_strategy: json

logging:
  verbosity: info
  track_annotations: true
  log_spans: true
```

## CLI Usage

### Basic Usage

```bash
# Process single sequence
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1

# Process multiple sequences
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1,5,10

# Process range of sequences
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100

# With custom configuration files
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100 \
  --config config/pipelines/span_annotator.yaml \
  --agent config/agents/span_annotator_agent.yaml

# Consolidate existing results only
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100 \
  --consolidate-only
```

### Advanced Usage

```bash
# Custom configuration files
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-50 \
  --config config/pipelines/span_annotator.yaml \
  --agent config/agents/span_annotator_agent.yaml

# Resume processing with specific range (automatic resume enabled)
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 51-100

# Consolidate existing results without processing new sequences
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100 \
  --consolidate-only
```

### Multi-Process Coordination

For large-scale annotation (50k+ sequences), run multiple processes on different ranges:

```bash
# Terminal 1: Process sequences 1-1000
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-1000

# Terminal 2: Process sequences 1001-2000
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1001-2000

# Terminal 3: Process sequences 2001-3000
uv run -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 2001-3000
```

The global `metadata.json` automatically tracks active processes and prevents conflicts.

## Integration with Training Pipeline

### Annotation Record Generation

Span annotations are packaged into `AnnotationRecord` objects for training:

```python
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation

# Create training record
record = AnnotationRecord(
    raw=sequence_text,
    sequence_id=sequence_number,
    embedding_chunk_id=chunk_id,
    span_annotations=[
        SpanAnnotation(
            start_pos=4,
            end_pos=19,
            xbar_class="NP",
            confidence=0.95
        )
    ],
    total_positions=len(sequence_text)
)
```

### Batch Processing with Resume Capability

The system supports efficient batch processing with resume capabilities:

```python
# Process large batches with automatic resume
async def process_corpus(corpus_path: str, output_path: str):
    pipeline = SpanAnnotatorPipeline()
    
    # Load existing results for resume
    existing_results = pipeline.load_existing_results(Path(output_path))
    
    # Process remaining sequences
    results = await pipeline.process_sequence_range(
        corpus_file=Path(corpus_path),
        output_dir=Path(output_path),
        range_spec="1-1000",
        resume=True
    )
    
    logger.info(f"Processed {results['successful_annotations']} sequences")
```

## Validation and Quality Control

### Position Validation

All position spans are validated against sequence boundaries:

```python
def validate_position_span(span: PositionSpan, sequence_length: int) -> bool:
    """Validate position span against sequence constraints."""
    return (
        0 <= span.start_pos < sequence_length and
        span.start_pos < span.end_pos <= sequence_length and
        span.end_pos - span.start_pos >= 1  # Minimum span length
    )
```

### Linguistic Coherence Checks

- **Whitespace Coherence**: Spans respect word boundaries when appropriate
- **Hierarchical Consistency**: Overlapping spans maintain proper nesting relationships
- **X-bar Compliance**: Generated spans follow X-bar theory constraints
- **JSON Validation**: All LLM responses are validated for proper JSON structure
- **Confidence Scoring**: LLM responses include confidence scores for span quality assessment

## Performance Characteristics

### Async Processing Benefits

- **Concurrent Sessions**: Multiple LLM conversations can run simultaneously (max_concurrent: 5)
- **I/O Efficiency**: Async file operations and network requests
- **Memory Management**: Streaming processing for large corpora
- **Fault Tolerance**: Individual session failures don't affect batch processing

### Scalability Considerations

- **Batch Size Tuning**: Configurable batch sizes (default: 64) based on available resources
- **Rate Limiting**: Built-in rate limiting for LLM API calls (request_delay: 0.5s)
- **Memory Usage**: Streaming approach keeps memory usage constant
- **Checkpointing**: Regular saves enable processing of very large corpora
- **Resume Capability**: Failed or interrupted runs can be resumed without losing progress

## Future Extensions

### Enhanced Linguistic Analysis

- **Dependency Relations**: Extend to capture syntactic dependencies
- **Semantic Roles**: Add semantic role labeling capabilities
- **Cross-lingual Support**: Multi-language annotation templates
- **Domain Specialization**: Custom classifiers for specific domains

### Training Integration

- **Active Learning**: Select most informative spans for annotation
- **Confidence Weighting**: Use annotation confidence in training loss
- **Adversarial Validation**: Cross-validate annotations between different agents
- **Human-in-the-Loop**: Integration with human annotation tools for quality control

### Performance Optimization

- **Caching**: Cache system prompts and common responses
- **Batch Optimization**: Dynamic batch size adjustment based on performance
- **Model Optimization**: Support for different model sizes and capabilities
- **Distributed Processing**: Support for distributed annotation across multiple machines
- Function definitions and calls
- Variable declarations and references  
- Control flow structures
- Expression boundaries

Code: "{text}"
"""
```

### Quality Focus and Error Handling

- **Malformed Response Handling**: Invalid JSON responses are logged but don't block processing
- **Confidence Scoring**: LLM responses include confidence scores for span quality assessment
- **Resumable Processing**: Failed sessions can be restarted without losing progress
- **Statistical Tracking**: Success/failure rates tracked per classifier type

## Integration with Training Pipeline

### Annotation Record Generation

Span annotations are packaged into `AnnotationRecord` objects for training:

```python
from x_spanformer.schema.annotation_record import AnnotationRecord, SpanAnnotation

# Create training record
record = AnnotationRecord(
    raw=sequence_text,
    sequence_id=sequence_number,
    embedding_chunk_id=chunk_id,
    span_annotations=[
        SpanAnnotation(
            start_pos=4,
            end_pos=19,
            xbar_class="NP",
            confidence=0.95
        )
    ],
    total_positions=len(sequence_text)
)
```

### Batch Processing with Resume Capability

The system supports efficient batch processing with resume capabilities:

```python
# Process large batches with automatic resume
async def process_corpus(corpus_path: str, output_path: str):
    agent = SpanAnnotatorSession()
    
    # Load existing results for resume
    existing_results = load_existing_annotations(output_path)
    
    # Process remaining sequences
    async for batch in agent.process_corpus_batch(
        corpus_path, 
        resume_from=existing_results,
        batch_size=32
    ):
        save_annotation_batch(batch, output_path)
        logger.info(f"Processed {len(batch.records)} sequences")
```

## Validation and Quality Control

### Position Validation

All position spans are validated against sequence boundaries:

```python
def validate_position_span(span: PositionSpan, sequence_length: int) -> bool:
    """Validate position span against sequence constraints."""
    return (
        0 <= span.start_pos < sequence_length and
        span.start_pos < span.end_pos <= sequence_length and
        span.end_pos - span.start_pos >= 1  # Minimum span length
    )
```

### Linguistic Coherence Checks

- **Whitespace Coherence**: Spans respect word boundaries when appropriate
- **Hierarchical Consistency**: Overlapping spans maintain proper nesting relationships
- **X-bar Compliance**: Generated spans follow X-bar theory constraints

## Performance Characteristics

### Async Processing Benefits

- **Concurrent Sessions**: Multiple LLM conversations can run simultaneously
- **I/O Efficiency**: Async file operations and network requests
- **Memory Management**: Streaming processing for large corpora
- **Fault Tolerance**: Individual session failures don't affect batch processing

### Scalability Considerations

- **Batch Size Tuning**: Configurable batch sizes based on available resources
- **Rate Limiting**: Built-in rate limiting for LLM API calls
- **Memory Usage**: Streaming approach keeps memory usage constant
- **Checkpointing**: Regular saves enable processing of very large corpora

## Future Extensions

### Enhanced Linguistic Analysis

- **Dependency Relations**: Extend to capture syntactic dependencies
- **Semantic Roles**: Add semantic role labeling capabilities
- **Cross-lingual Support**: Multi-language annotation templates
- **Domain Specialization**: Custom classifiers for specific domains

### Training Integration

- **Active Learning**: Select most informative spans for annotation
- **Confidence Weighting**: Use annotation confidence in training loss
- **Adversarial Validation**: Cross-validate annotations between different agents
- **Human-in-the-Loop**: Integration with human annotation tools for quality control
