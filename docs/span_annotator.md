# Span Annotator Agent

## Overview

The X-Spanformer span annotator agent implements a comprehensive annotation pipeline for generating hierarchical X-bar span labels from raw Unicode sequences. This pipeline uses multi-turn agentic conversations to create training data for the factorized pointer network boundary prediction system.

## Architecture

### Bidirectional Contextual Foundation

The annotation system is built on X-Spanformer's bidirectional contextual embedding architecture:

- **Position Embeddings**: Each H[t] ∈ R^512 represents bidirectional contextual character information
- **Multi-scale Context**: Dilated convolutions capture patterns at multiple scales with full sequence awareness
- **Boundary Detection**: Start and end positions access both preceding and following contextual information
- **No Span Averaging**: Spans detected through boundary prediction, not embedding aggregation

### Async Agent Pattern

The `SpanAnnotatorAgent` follows established async patterns from the agents package:

```python
from x_spanformer.agents.span_annotator import SpanAnnotatorAgent
from x_spanformer.agents.dialogue import DialogueManager
from x_spanformer.agents.position_mapper import PositionMapper

# Initialize agent with dialogue management
agent = SpanAnnotatorAgent(
    model_name="phi4-mini",
    dialogue_manager=DialogueManager()
)

# Process sequences asynchronously
async for result in agent.annotate_batch(sequences):
    # Handle span annotations with position alignment
    for annotation in result.span_annotations:
        print(f"Span: {annotation.start_pos}-{annotation.end_pos}")
```

## Position-wise Alignment

### Character-to-Position Mapping

The system uses the `PositionMapper` to convert character-level spans to position-wise indices:

```python
from x_spanformer.agents.position_mapper import PositionMapper

mapper = PositionMapper()

# Character-level span from LLM agent
char_span = CharacterSpan(
    start_char=4,
    end_char=18,
    xbar_class="NP",
    text="quick brown fox"
)

# Convert to position-wise for training
pos_span = mapper.char_to_position(char_span, original_text)
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

## Multi-turn Conversation Strategy

### Domain-Aware Processing

The agent uses domain-specific templates for natural language, code, and mixed content:

```python
# Natural language processing
natural_prompt = """
Analyze the following text for syntactic spans using X-bar theory.
Identify noun phrases, verb phrases, and clauses with their boundaries.

Text: "{text}"

Return spans in JSON format with start/end character indices.
"""

# Code analysis
code_prompt = """
Analyze the following code for structural spans including:
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
    agent = SpanAnnotatorAgent()
    
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
