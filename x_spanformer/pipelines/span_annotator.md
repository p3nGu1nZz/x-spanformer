# Span Annotator Pipeline

## Overview

The Unified Span Annotator Pipeline is a **production-ready, battle-tested** implementation that generates high-quality training data for X-Spanformer's factorized pointer network boundary predictor. It combines three-turn hierarchical annotation with robust async session management, comprehensive error handling, and enhanced JSON parsing robustness.

## Key Features

- **Three-turn conversation strategy**: Progressive analysis from word-level → phrase-level → clause-level
- **Enhanced JSON parsing robustness**: Handles truncated LLM responses, malformed JSON, and case-insensitive text matching
- **Advanced label cleaning system**: Comprehensive word span validation with pattern-based filtering
- **Resume capability**: Automatically resumes from previous progress with gap detection
- **Production-grade error handling**: Comprehensive span validation and position verification
- **Intelligent logging system**: Aggregated counts instead of verbose repetitive messages
- **Real-time telemetry**: Progress tracking with detailed span type statistics
- **Multiple output formats**: Working files, consolidated results, and analysis reports
- **Factorized pointer network ready**: Generates training data perfectly aligned with Section 3.3 architecture

## Architecture

### Core Components

1. **SpanAnnotatorPipeline**: Main pipeline orchestrator with resume and gap detection
2. **SpanAnnotatorSession**: Async session management with timeout controls
3. **XBarAnnotator**: X-bar theory-based span extraction with enhanced JSON parsing
4. **XBarLabelMap**: Advanced label cleaning and word span validation system
5. **Output Management**: Working files, consolidation, metadata, and analysis reports
6. **JSON Parsing Robustness**: Truncation detection, malformed JSON recovery, case-insensitive matching

### Processing Flow

```
Input (corpus.jsonl) → Load Sequences → Filter by Range → 
Process in Batches → Annotate with XBar → Enhanced JSON Parsing → 
Save Working Files → Position Validation → Label Cleaning & Word Span Validation → 
Consolidate Results → Generate Metadata → Analysis Reports → Output
```

### Production Status (August 2025)

**✅ PRODUCTION READY**: Successfully processing sequences with zero position errors and advanced label cleaning
- **60,558 clean annotations** from 61,053 original spans (99.2% retention rate)
- **495 invalid word spans** automatically filtered using pattern-based validation
- **352 labels mapped** from invalid to valid categories with aggregated logging
- **Zero validation errors** in position encoding and text extraction
- **Enhanced JSON robustness** handling truncated LLM responses
- **Intelligent logging system** with count aggregation instead of repetitive debug messages
- **Comprehensive word span validation** supporting percentages, abbreviations, and expressions
- **Perfect alignment** with factorized pointer network requirements (Section 3.3)

## Usage

### Command Line Interface

```bash
# Basic usage with improved logging
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --output data/annotations \
    --range 1-100 \
    --model llama3.2:3b \
    --temperature 0.2 \
    --timeout 180.0 \
    --verbose \
    --stream

# Parallel processing for large-scale annotation
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --range 1-1000 &
python -m x_spanformer.pipelines.span_annotator \
    --corpus data/vocab/corpus.jsonl \
    --range 1001-2000 &
```

### Enhanced Logging and Output

#### Improved Sequence Selection Logging (August 2025)

The pipeline now features concise, informative logging that replaces verbose sequence lists with summary information:

```
2025-08-08 15:01:41,608 - __main__ - INFO - Selected 1000 sequences (1 to 1000) out of 1000 requested
2025-08-08 15:01:41,609 - __main__ - INFO - Filtered to 1000/5179 sequences
```

**Benefits:**
- **Readable logs**: No more thousand-line sequence dumps
- **Essential information**: Shows count, range, and success rate
- **Performance**: Faster logging with reduced I/O overhead
- **Debugging**: Maintains all critical filtering information

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
| `--stream` | flag | False | Stream results to console in real-time |

### API Integration

The span annotator integrates with X-Spanformer's core components through the **Ollama Client API**:

#### Ollama Client Interface

```python
# Core chat function for LLM communication
async def chat(
    model: str,
    conversation: List[Message],
    system: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 60.0
) -> str
```

**Features:**
- **Async communication**: Non-blocking LLM interactions
- **Conversation history**: Multi-turn context preservation
- **Temperature control**: Creativity vs consistency tuning
- **Timeout management**: Prevents hanging on slow responses
- **Error handling**: Comprehensive connection and response error recovery

## X-Spanformer Paper Alignment

This pipeline implements **Section 3.3: Span Predictor** from the X-Spanformer paper, generating training data for the factorized pointer network boundary predictor.

### Theoretical Foundation

**X-bar Theory Integration**: The annotation process follows X-bar linguistic theory for hierarchical phrase structure:
- **Terminal nodes** (X⁰): Word-level categories (noun, verb, adjective, etc.)
- **Intermediate projections** (X'): Phrase-level structures (noun_phrase, verb_phrase, etc.) 
- **Maximal projections** (XP): Clause-level constructions (main_clause, subordinate_clause, etc.)

### Architecture Alignment

```
Raw Text → X-bar Annotation → Boundary Positions → 
Factorized Pointer Network Training Data
```

**Paper Section 3.3 Implementation:**
- **Factorized boundary prediction**: Independent start/end position classification
- **Multi-label support**: Overlapping spans at different hierarchical levels
- **Position-wise encoding**: Binary classification targets for each text position
- **BCE loss optimization**: Sigmoid-normalized boundary probabilities

### Training Data Format

The pipeline generates training targets perfectly aligned with the paper's factorized pointer network:

```python
# Start position targets (binary classification)
start_targets = torch.zeros(sequence_length, dtype=torch.float32)
start_targets[span_start_positions] = 1.0

# End position targets (binary classification)  
end_targets = torch.zeros(sequence_length, dtype=torch.float32)
end_targets[span_end_positions] = 1.0
```

This format enables **independent boundary prediction** as described in Section 3.3, where start and end positions are predicted by separate linear heads rather than joint span classification.

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
    stream=True
)
```

## Recent Improvements (August 2025)

### Logging Enhancements

**Concise Sequence Selection**: Replaced verbose sequence lists with informative summaries:
- **Before**: `Selected sequences: [1, 2, 3, ..., 1000]` (massive log output)
- **After**: `Selected 1000 sequences (1 to 1000) out of 1000 requested` (single line)

**Benefits:**
- **Performance**: Reduced I/O overhead and log file size
- **Readability**: Essential information without clutter
- **Debugging**: Maintains all critical filtering statistics
- **Scalability**: Handles large sequence ranges without log bloat

### Production Validation

**Zero-Error Processing**: Latest runs demonstrate production readiness:
- **1,703 spans** across 56 sequences with perfect position alignment
- **128.2% overlap ratio** supporting multi-label boundary prediction  
- **Enhanced JSON robustness** handling truncated LLM responses
- **Automatic recovery** from parsing errors at runtime

### Performance Optimizations

**Enhanced Error Handling**: Comprehensive recovery mechanisms:
- **Truncation detection**: Automatic identification of incomplete JSON responses
- **Malformed JSON recovery**: Robust parsing with fallback strategies
- **Case-insensitive matching**: Flexible text extraction and validation
- **Position verification**: 100% accuracy in span boundary calculation

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

Individual sequence results in `working/` directory with enhanced metadata:

```json
{
    "sequence_number": 1,
    "raw_text": "The quick brown fox jumps over the lazy dog.",
    "domain_type": "natural",
    "timestamp": "2025-08-08T02:53:04.722491",
    "status": "completed",
    "error_message": null,
    "span_annotations": [
        {
            "start_pos": 0,
            "end_pos": 3,
            "xbar_label": "determiner",
            "text": "The"
        },
        {
            "start_pos": 4,
            "end_pos": 9,
            "xbar_label": "adjective", 
            "text": "quick"
        },
        {
            "start_pos": 4,
            "end_pos": 19,
            "xbar_label": "noun_phrase",
            "text": "quick brown fox"
        },
        {
            "start_pos": 0,
            "end_pos": 43,
            "xbar_label": "main_clause",
            "text": "The quick brown fox jumps over the lazy dog"
        }
    ],
    "total_spans": 15,
    "agent_metadata": {
        "strategy": "three_turn_xbar",
        "model": "llama3.2:3b",
        "domain": "natural",
        "total_turns": 3,
        "word_spans": 9,
        "phrase_spans": 4,
        "clause_spans": 2,
        "total_valid_spans": 15
    }
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

## Enhanced JSON Parsing Robustness

### Production Challenges Solved (August 2025)

The pipeline has been enhanced with comprehensive JSON parsing robustness to handle real-world LLM response variations:

#### Common LLM Response Issues
- **Truncated JSON responses**: `[{"text":"ti:j"}}` → Pipeline hanging
- **Malformed JSON arrays**: Missing brackets, extra commas, incomplete objects
- **Case sensitivity**: LLM returning `Text` instead of `text` in field names
- **Incomplete responses**: Partial JSON due to model timeout or token limits

#### Robustness Solutions Implemented

**1. Truncation Detection**
```python
def _detect_truncation(self, response: str) -> bool:
    """Detect truncated JSON responses that cause parser hangs."""
    if not response.strip():
        return True
    
    # Check for incomplete JSON structures
    open_brackets = response.count('[') + response.count('{')
    close_brackets = response.count(']') + response.count('}')
    
    return open_brackets > close_brackets
```

**2. Malformed JSON Recovery**
```python
def _fix_malformed_json(self, json_str: str) -> str:
    """Fix common JSON malformation patterns."""
    # Fix missing closing brackets
    # Remove trailing commas
    # Escape unescaped quotes
    # Handle incomplete objects
```

**3. Case-Insensitive Field Matching**
```python
def _extract_text_boundaries(self, text: str, target_text: str) -> Optional[Tuple[int, int]]:
    """Extract boundaries with case-insensitive matching."""
    # Handle case variations in LLM responses
    # Fuzzy matching for robust text extraction
```

#### Production Results
- **Zero parsing errors** across 1,703 spans in 56 sequences
- **100% position validation** success rate
- **Automatic recovery** from truncated responses at sequence 40
- **Enhanced reliability** for large-scale annotation tasks

## Enhanced Label Cleaning System

### Advanced Word Span Validation

The pipeline includes a comprehensive label cleaning and word span validation system that ensures high-quality training data:

#### Pattern-Based Word Span Filtering
- **Spaces detection**: Automatically removes spans containing spaces (not word-level)
- **Mixed character validation**: Filters invalid combinations of letters, numbers, and special characters
- **Identifier patterns**: Allows valid programming identifiers (letters + underscores/hyphens)
- **Number formats**: Supports integers, decimals, negative numbers, and percentages
- **Abbreviations**: Allows words with periods (e.g., "Dr.", "U.S.", "etc.")
- **Expressions**: Supports bracketed `[83]`, parenthetical `(t)`, and pipe `|s|` expressions
- **Trailing punctuation**: Allows words ending with colons ("words:")

#### Label Mapping System
- **Intelligent mapping**: Converts invalid labels to valid X-bar categories
- **Aggregated logging**: Shows mapping counts instead of repetitive debug messages
- **Statistical reporting**: Provides comprehensive cleaning statistics
- **Zero data loss**: Maps rather than removes when possible

#### Production Cleaning Results (August 2025)
```
Label cleaning results:
  Valid labels (unchanged): 60,206
  Invalid labels mapped: 352
  Invalid labels removed: 0
  Invalid word spans removed: 495
  Total annotations before cleaning: 61,053
  Total annotations after cleaning: 60,558
  Total spans filtered: 495
```

**Key Statistics:**
- **99.2% retention rate** - Only 495 spans (0.8%) filtered for quality
- **Zero label removal** - All invalid labels successfully mapped to valid categories
- **352 labels mapped** from variations like "proper noun" → "noun", "auxiliary" → "verb"
- **Clean logging output** - Aggregated counts replace thousands of repetitive debug messages

#### Supported Word Span Patterns
- **Pure text**: `"transformer"`, `"attention"`
- **Numbers**: `"42"`, `"3.14"`, `"-5"`, `"2.7%"`
- **Identifiers**: `"attention_weights"`, `"multi-head"`
- **Abbreviations**: `"Dr."`, `"U.S."`, `"etc."`
- **Expressions**: `"[83]"`, `"(t)"`, `"|s|"`
- **Punctuated**: `"words:"`, `"Note:"`
- **Version numbers**: `"1.2.3"`, `"v2.0"`

## X-bar Theory Integration

### Linguistic Foundation

The pipeline uses X-bar theory for syntactic analysis, providing hierarchical span classification:

- **Word Level**: Basic lexical categories (N, V, Adj, Adv, D, P, etc.)
- **Phrase Level**: Intermediate projections (N', V', etc.)
- **Clause Level**: Maximal projections (NP, VP, CP, etc.)

### Domain-Specific Classification (Production Output)

#### Natural Language Domain (23.9% noun, 8.3% adjective, 7.0% verb)
- **Word**: noun, verb, adjective, adverb, determiner, preposition, conjunction, pronoun
- **Phrase**: noun_phrase, verb_phrase, expression, documentation_comment  
- **Clause**: main_clause, subordinate_clause, conditional_clause

#### Code Domain (5.9% identifier, 7.3% literal, 3.8% operator)
- **Word**: keyword, identifier, literal, operator, delimiter
- **Phrase**: code_block, function_definition, method_call, expression
- **Statement**: assignment, function_call, control_flow, declaration

#### Mixed Domain (Multi-modal text with 2.7% code_block)
- **Technical elements**: identifier, literal, operator seamlessly integrated
- **Natural language**: noun, verb, adjective maintaining linguistic structure
- **Special structures**: documentation_comment, code_block, function_definition
- **Hierarchical organization**: word → phrase → clause across modalities

#### Production Span Distribution (August 2025)
- **Word-level spans**: 1,211 (71.1%) - avg length 5.5 characters
- **Phrase-level spans**: 244 (14.3%) - avg length 28.1 characters  
- **Clause-level spans**: 142 (8.3%) - avg length 47.1 characters
- **Other spans**: 106 (6.2%) - specialized categories
- **Total overlap ratio**: 128.2% (multi-label boundary support)

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
