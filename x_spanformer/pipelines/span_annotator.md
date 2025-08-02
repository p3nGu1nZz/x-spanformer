# Span Annotator Pipeline

## Overview

The `span_annotator` pipeline implements **agentic X-bar span annotation** for generating supervised training data for the factorized pointer network span predictor. This pipeline processes raw Unicode sequences from corpus.jsonl through multi-turn conversations with phi4-mini to generate hierarchical X-bar span boundary annotations.

### Key Features

- **Comprehensive Classifier Extraction**: Processes each sequence for ALL applicable xbar.py classifiers
- **Domain-Aware Processing**: Automatic domain detection (natural/code/mixed) with tailored prompts
- **Agent Pattern Integration**: Uses judge_session.py, ollama_client.py, and dialogue.py patterns
- **Scalable Architecture**: Designed for 50k+ LLM requests with statistical tracking
- **Resumable Processing**: Individual working files enable inspection and continuation
- **Multi-Process Support**: Parallel processing on different sequence ranges

## Architecture

### SpanAnnotatorSession Class

The core annotation engine follows async conversation patterns from the agents package:

```python
class SpanAnnotatorSession:
    """Async X-bar span annotation session using comprehensive conversation approach."""
    
    def __init__(self, config_file="span_annotator.yaml"):
        self.cfg = load_config(config_file)
        self.jinja_env = setup_jinja_templates()
        self.ollama_client = AsyncOllamaClient(self.cfg.model)
        self.dialogue = Dialogue()  # Multi-turn conversation management
        
    async def extract_all_classifiers(self, raw_sequence: str, corpus_id: str, domain_type: str) -> dict:
        """Extract all applicable classifiers through single comprehensive conversation."""
        
        # Create comprehensive system prompt with ALL classifier info
        system_prompt = self._build_comprehensive_system_prompt(domain_type)
        classifier_map = self._get_xbar_classifier_map(domain_type)
        
        # Single conversation covering all classifiers
        conversation_result = await self._conduct_comprehensive_annotation(
            raw_sequence, system_prompt, classifier_map
        )
        
        return self._parse_comprehensive_response(conversation_result, classifier_map)
        
    def _build_comprehensive_system_prompt(self, domain_type: str) -> str:
        """Build system prompt containing ALL xbar classifier definitions for domain."""
        
    def _get_xbar_classifier_map(self, domain_type: str) -> Dict[str, str]:
        """Return mapping of all applicable xbar classifiers to their descriptions."""
        
    async def _conduct_comprehensive_annotation(self, sequence: str, system_prompt: str, classifiers: Dict) -> dict:
        """Conduct single multi-turn conversation covering all classifiers."""
```
```

### Agent Pattern Integration

Following the established async patterns from the agents package:

1. **AsyncOllamaClient Integration**: Async integration with `ollama_client.py` for model requests
2. **Comprehensive System Prompts**: Single conversation with complete xbar classifier mapping
3. **Dialogue Management**: Multi-turn conversation handling via `dialogue.py` patterns
4. **Configuration Management**: YAML-based configuration following pipeline conventions

### XBar Classifier Mapping Integration

Instead of individual template files, the pipeline uses comprehensive classifier mapping:

```python
# x_spanformer/agents/xbar_map.py - comprehensive classifier definitions

NATURAL_CLASSIFIERS = {
    "noun": "Identify all nouns including proper nouns, common nouns, and collective nouns",
    "verb": "Identify all verbs including action verbs, linking verbs, and auxiliary verbs",
    "adjective": "Identify all adjectives including descriptive, comparative, and superlative forms",
    "noun_phrase": "Identify noun phrases including determiners, modifiers, and head nouns",
    "verb_phrase": "Identify verb phrases including auxiliary verbs, main verbs, and complements",
    # ... comprehensive mapping for all natural language classifiers
}

CODE_CLASSIFIERS = {
    "keyword": "Identify programming language keywords (if, for, class, def, etc.)",
    "identifier": "Identify variable names, function names, and class names",
    "function_call": "Identify function call expressions with arguments",
    "assignment": "Identify variable assignment statements",
    # ... comprehensive mapping for all code classifiers
}

MIXED_CLASSIFIERS = {
    "inline_code": "Identify inline code snippets within natural language text",
    "code_block": "Identify code blocks or examples within documentation",
    "natural_instruction": "Identify natural language instructions about code",
    # ... comprehensive mapping for mixed domain classifiers
}
```

## Processing Strategy

### Comprehensive Sequence Processing

Each sequence undergoes systematic processing through a single comprehensive conversation:

1. **Domain Detection**: Automatic detection from corpus.jsonl type field
2. **Classifier Map Building**: Build complete xbar classifier mapping for domain
3. **System Prompt Construction**: Create comprehensive prompt with ALL classifier definitions
4. **Single Async Conversation**: One multi-turn conversation covering all applicable classifiers
5. **Response Parsing**: Parse comprehensive JSON response and validate span boundaries
6. **Statistics Tracking**: Track success/failure rates per classifier type

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
    
    # Parse range specification (e.g., "1-100", "1,5,10", "42")
    sequence_ids = parse_range_specification(sequence_range)
    
    # Load and filter corpus to target sequences
    target_sequences = load_target_sequences(corpus_file, sequence_ids)
    
    # Process each sequence comprehensively
    session = SpanAnnotatorSession()
    for record in target_sequences:
        working_file = output_dir / "working" / f"{record.corpus_id}.json"
        
        if working_file.exists():
            # Resume from existing working file
            sequence_data = json.load(working_file.open())
            if sequence_data.get("annotation_status") == "completed":
                continue  # Skip completed sequences
        
        # Single comprehensive annotation conversation
        try:
            annotation_result = await session.extract_all_classifiers(
                record.raw, record.corpus_id, record.domain_type
            )
            
            # Save comprehensive results to working file
            save_annotation_result(working_file, record, annotation_result)
            
            # Update global metadata
            update_global_metadata(output_dir / "metadata.json", record.corpus_id, annotation_result)
            
        except Exception as e:
            # Log failed annotation and continue
            log_annotation_failure(working_file, record, str(e))
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
    "status": "keep",
    "source": "consolidated_corpus"
  },
  "annotation_session": {
    "started_at": "2025-08-01T15:30:00Z",
    "completed_at": "2025-08-01T15:31:45Z",
    "annotation_status": "completed",
    "model": "phi4-mini",
    "conversation_turns": 3,
    "total_processing_time": 105.2,
    "classifiers_attempted": 23,
    "successful_extractions": 21,
    "failed_extractions": 2
  },
  "comprehensive_annotations": {
    "all_spans": [
      {
        "text": "The",
        "char_start": 0,
        "char_end": 2,
        "xbar_label": "determiner",
        "xbar_role": "specifier",
        "hierarchical_level": "word",
        "confidence": 0.95
      },
      {
        "text": "quick brown fox",
        "char_start": 4,
        "char_end": 18,
        "xbar_label": "noun_phrase", 
        "xbar_role": "subject",
        "hierarchical_level": "phrase",
        "confidence": 0.88
      }
    ],
    "classifier_coverage": {
      "noun": {"attempted": true, "success": true, "spans_found": 3},
      "verb": {"attempted": true, "success": true, "spans_found": 1},
      "determiner": {"attempted": true, "success": true, "spans_found": 2},
      "noun_phrase": {"attempted": true, "success": true, "spans_found": 2},
      "verb_phrase": {"attempted": true, "success": true, "spans_found": 1},
      "adjective": {"attempted": true, "success": false, "error": "No clear adjective spans identified"}
    }
  },
  "conversation_log": [
    {
      "role": "system",
      "content": "You are an expert linguistic annotator specializing in X-bar theory span identification..."
    },
    {
      "role": "user", 
      "content": "Analyze this sequence for ALL applicable natural language X-bar spans: 'The quick brown fox jumps over the lazy dog.'"
    },
    {
      "role": "assistant",
      "content": "I'll provide comprehensive X-bar span annotations for this sentence. Let me identify spans at all hierarchical levels..."
    }
  ]
}
    },
    "natural_roles": {
      "subject": {
        "status": "success",
        "attempted_at": "2025-08-01T15:33:00Z",
        "spans_extracted": 1,
        "template_used": "natural/roles/subject.j2",
        "model_response": "{\"spans\": [{\"text\": \"The quick brown fox\", \"start\": 0, \"end\": 18, \"role\": \"subject\"}]}"
      }
    },
    "xbar_roles": {
      "head": {
        "status": "failed",
        "attempted_at": "2025-08-01T15:34:00Z",
        "error": "Invalid JSON response",
        "template_used": "xbar/roles/head.j2",
        "model_response": "I found several head elements: fox (core noun), jumps (main verb)"
      }
    }
  },
  "annotations": [
    {
      "char_start": 16,
      "char_end": 18,
      "span_text": "fox",
      "xbar_label": "noun",
      "xbar_role": "head",
      "hierarchical_level": "word",
      "extraction_metadata": {
        "classifier": "natural_labels.noun",
        "confidence": 0.95,
        "template": "natural/labels/noun.j2"
      }
    },
    {
      "char_start": 20,
      "char_end": 24,
      "span_text": "jumps",
      "xbar_label": "verb",
      "xbar_role": "predicate",
      "hierarchical_level": "word",
      "extraction_metadata": {
        "classifier": "natural_labels.verb",
        "confidence": 0.98,
        "template": "natural/labels/verb.j2"
      }
    }
  ],
  "failed_requests": [
    {
      "classifier": "xbar_roles.head",
      "error": "Invalid JSON response",
      "template": "xbar/roles/head.j2",
      "model_response": "I found several head elements: fox (core noun), jumps (main verb)",
      "attempted_at": "2025-08-01T15:34:00Z"
    }
  ]
}
```

### Global Metadata Format

```json
{
  "pipeline_version": "1.0",
  "started_at": "2025-08-01T15:00:00Z",
  "last_updated": "2025-08-01T16:45:00Z",
  "corpus_file": "data/vocab/corpus.jsonl",
  "output_directory": "data/annotations",
  "total_sequences": 5107,
  "processed_sequences": 1203,
  "remaining_sequences": 3904,
  "total_classifier_attempts": 43208,
  "successful_extractions": 28941,
  "failed_extractions": 14267,
  "classifier_statistics": {
    "natural_labels": {
      "noun": {"attempts": 1567, "successes": 1234, "success_rate": 0.787, "total_spans": 3456},
      "verb": {"attempts": 1567, "successes": 1345, "success_rate": 0.858, "total_spans": 2234},
      "adjective": {"attempts": 1567, "successes": 987, "success_rate": 0.630, "total_spans": 1876}
    },
    "code_labels": {
      "keyword": {"attempts": 876, "successes": 789, "success_rate": 0.901, "total_spans": 1234},
      "identifier": {"attempts": 876, "successes": 823, "success_rate": 0.940, "total_spans": 2345},
      "operator": {"attempts": 876, "successes": 734, "success_rate": 0.838, "total_spans": 987}
    },
    "xbar_roles": {
      "head": {"attempts": 5107, "successes": 3456, "success_rate": 0.677, "total_spans": 4567},
      "specifier": {"attempts": 5107, "successes": 2134, "success_rate": 0.418, "total_spans": 2876},
      "modifier": {"attempts": 5107, "successes": 1876, "success_rate": 0.367, "total_spans": 3234}
    }
  },
  "domain_breakdown": {
    "natural": {"sequences": 2567, "total_spans": 34567, "avg_spans_per_sequence": 13.47},
    "code": {"sequences": 1876, "total_spans": 18923, "avg_spans_per_sequence": 10.09},
    "mixed": {"sequences": 664, "total_spans": 8234, "avg_spans_per_sequence": 12.40}
  },
  "training_dataset_progress": {
    "current_total_spans": 61724,
    "target_spans": 100000,
    "completion_rate": 0.617,
    "estimated_sequences_needed": 1956,
    "priority_classifiers": ["noun", "verb", "identifier", "keyword", "head"]
  },
  "active_processes": [
    {"pid": 1234, "range": "1-100", "started_at": "2025-08-01T16:30:00Z", "progress": "85/100"},
    {"pid": 1235, "range": "101-200", "started_at": "2025-08-01T16:35:00Z", "progress": "23/100"}
  ]
}
```

## Configuration

### Default Configuration (span_annotator.yaml)

```yaml
# Model configuration
model:
  name: "phi4-mini"
  temperature: 0.1
  max_tokens: 1024
  timeout: 30

# Agent configuration
agent:
  max_retries: 3
  retry_delay: 1.0
  template_directory: "templates"
  response_format: "json"

# Processing configuration
processing:
  chunk_size: 100
  max_parallel_requests: 4
  request_delay: 0.5
  max_sequence_length: 512

# xbar classifier configuration
classifiers:
  natural_labels: ["noun", "verb", "adjective", "adverb", "determiner", "preposition", "pronoun", "conjunction", "punctuation"]
  natural_roles: ["subject", "predicate", "modifier", "complement"]
  code_labels: ["keyword", "identifier", "operator", "literal", "delimiter", "type_name", "comment"]
  code_roles: ["function", "declaration", "expression", "statement"]
  mixed_roles: ["inline_code", "documentation_context", "instructional_sequence"]
  xbar_roles: ["head", "specifier", "modifier", "complement", "adjunct"]

# Output configuration
output:
  save_working_files: true
  save_failed_requests: true
  consolidate_on_completion: true
  include_metadata: true

# Validation configuration
validation:
  max_char_start: 512
  max_char_end: 512
  min_span_length: 1
  max_span_length: 100
  require_valid_json: true
```

## CLI Usage

### Basic Usage

```bash
# Process single sequence
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1

# Process multiple sequences
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1,5,10

# Process range of sequences
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100
```

### Advanced Usage

```bash
# Custom configuration
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --config config/custom_span_annotator.yaml \
  --range 1-50

# Resume processing with specific range
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 51-100

# High-throughput processing
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-1000 \
  --config config/high_throughput_span_annotator.yaml
```

### Multi-Process Coordination

For large-scale annotation (50k+ sequences), run multiple processes on different ranges:

```bash
# Terminal 1: Process sequences 1-1000
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-1000

# Terminal 2: Process sequences 1001-2000
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1001-2000

# Terminal 3: Process sequences 2001-3000
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 2001-3000
```

The global `metadata.json` automatically tracks active processes and prevents conflicts.

## Comprehensive Annotation System

### XBar Classifier Mapping

Instead of individual templates, the pipeline uses comprehensive classifier definitions:

```python
# Example comprehensive system prompt construction
def build_comprehensive_system_prompt(domain_type: str) -> str:
    """Build system prompt with ALL xbar classifiers for domain."""
    
    base_prompt = """
You are an expert linguistic annotator specializing in X-bar theory span identification.
Your task is to identify ALL applicable linguistic spans in the given text using the comprehensive classifier set below.

IMPORTANT GUIDELINES:
1. Use position-based indexing (character positions in the original text)
2. Use 0-based indexing with char_end INCLUSIVE (last character of span)
3. Provide confidence scores between 0.0 and 1.0
4. Ensure spans don't inappropriately overlap
5. Include hierarchical levels: word, phrase, clause, sentence
6. Return comprehensive JSON with ALL identified spans

"""
    
    if domain_type == "natural":
        classifier_definitions = """
NATURAL LANGUAGE CLASSIFIERS:

Word Level:
- noun: Identify all nouns (proper, common, collective)
- verb: Identify all verbs (action, linking, auxiliary)
- adjective: Identify all adjectives (descriptive, comparative, superlative)
- adverb: Identify all adverbs (manner, time, place, degree)
- determiner: Identify determiners (the, a, an, this, that, these, those)
- preposition: Identify prepositions (in, on, at, by, for, with, etc.)

Phrase Level:
- noun_phrase: Identify complete noun phrases with modifiers
- verb_phrase: Identify complete verb phrases with complements
- adjective_phrase: Identify adjective phrases with modifiers
- prepositional_phrase: Identify prepositional phrases

Clause Level:
- main_clause: Identify main/independent clauses
- subordinate_clause: Identify dependent/subordinate clauses

Sentence Level:
- simple_sentence: Identify complete simple sentences
- compound_sentence: Identify compound sentences with coordination
"""
    
    elif domain_type == "code":
        classifier_definitions = """
CODE CLASSIFIERS:

Word Level:
- keyword: Programming language keywords (if, for, class, def, return, etc.)
- identifier: Variable names, function names, class names
- operator: All operators (+, -, *, /, ==, !=, &&, ||, etc.)
- literal: String literals, numeric literals, boolean literals

Phrase Level:
- expression: Mathematical or logical expressions
- function_call: Function calls with arguments
- assignment: Variable assignment statements
- parameter_list: Function parameter lists
- argument_list: Function call argument lists

Statement Level:
- if_statement: Conditional statements
- loop_statement: For loops, while loops
- function_definition: Complete function definitions
- class_definition: Complete class definitions
"""
    
    return base_prompt + classifier_definitions
```

## Shared Utilities Integration

The pipeline leverages existing shared utilities and introduces new ones:

### Existing Utilities
- **jsonl_processor.py**: For loading and validating corpus.jsonl files
- **text_processor.py**: For text splitting and length management if needed
- **position_mapper.py**: For character-to-position alignment in tokenizer-free architecture

### New Shared Utilities

```python
# x_spanformer/agents/xbar_map.py - comprehensive classifier mappings

class XBarClassifierMap:
    """Comprehensive X-bar classifier definitions for all domains."""
    
    def get_natural_classifiers(self) -> Dict[str, str]:
        """Return all natural language classifiers with descriptions."""
        
    def get_code_classifiers(self) -> Dict[str, str]:
        """Return all code classifiers with descriptions."""
        
    def get_mixed_classifiers(self) -> Dict[str, str]:
        """Return all mixed domain classifiers with descriptions."""
        
    def build_comprehensive_prompt(self, domain_type: str) -> str:
        """Build comprehensive system prompt for domain."""

# x_spanformer/pipelines/shared/annotation_processor.py

class ComprehensiveAnnotationProcessor:
    """Process comprehensive annotation responses from async conversations."""
    
    def parse_comprehensive_response(self, response: str, expected_classifiers: List[str]) -> Dict:
        """Parse model response containing all classifier annotations."""
        
    def validate_span_boundaries(self, spans: List[Dict], text_length: int) -> List[Dict]:
        """Validate and filter span boundaries for consistency."""
        
    def align_with_position_embeddings(self, spans: List[Dict], text: str) -> List[Dict]:
        """Align character spans with position-wise embeddings using PositionMapper."""

class AnnotationProcessor:
    """Shared utilities for annotation pipelines."""
    
    def validate_span_boundaries(self, sequence: str, start: int, end: int) -> bool:
        """Validate span boundaries are within sequence bounds."""
        
    def parse_json_response(self, response: str) -> List[dict]:
        """Parse and validate JSON response from LLM."""
        
    def consolidate_working_files(self, working_dir: Path, output_file: Path):
        """Consolidate individual working files into training JSONL."""
        
    def update_global_metadata(self, metadata_file: Path, updates: dict):
        """Thread-safe global metadata updates."""
```

## Performance Characteristics

### Computational Complexity

- **Single Sequence**: O(C) where C is number of applicable classifiers (~20-30)
- **Template Rendering**: O(1) per classifier (Jinja template compilation cached)
- **LLM Requests**: O(C) network requests per sequence
- **JSON Parsing**: O(R) where R is response length
- **File I/O**: O(1) per sequence (individual working files)

### Memory Usage

- **Working Memory**: ~1-2MB per sequence (JSON data + metadata)
- **Template Cache**: ~100KB for all Jinja templates
- **Global Metadata**: ~1-10MB depending on corpus size
- **Process Memory**: ~50-100MB per span_annotator process

### Scalability Characteristics

**Single Process**:
- **Throughput**: ~10-20 sequences/minute (depending on model speed)
- **Memory**: ~100MB baseline + working files
- **Network**: ~20-30 requests/minute to Ollama

**Multi-Process (N processes)**:
- **Throughput**: ~N × (10-20) sequences/minute
- **Memory**: N × 100MB + shared working directory
- **Network**: N × (20-30) requests/minute to Ollama
- **Coordination**: Lock-free through range-based assignment

### Expected Performance for 5k Sequences

**Sequential Processing**:
- **Total Time**: ~4-8 hours (assuming 10-20 sequences/minute)
- **Total Requests**: ~100k-150k LLM requests
- **Storage**: ~500MB-1GB for working files + consolidated output

**Parallel Processing (4 processes)**:
- **Total Time**: ~1-2 hours
- **Total Requests**: Same (~100k-150k)
- **Storage**: Same (~500MB-1GB)
- **Coordination**: Automatic through range assignment

## Resume and Validation

### Resume Capabilities

The pipeline includes comprehensive resume functionality:

1. **Working File Detection**: Automatically detects existing working files
2. **Progress Analysis**: Determines completed vs remaining classifiers per sequence
3. **Selective Processing**: Only processes missing classifiers
4. **Metadata Continuity**: Maintains statistics across resume sessions

### Validation Features

1. **Span Boundary Validation**: Ensures spans are within sequence bounds
2. **JSON Response Validation**: Validates LLM responses against expected schema
3. **Classifier Coverage**: Tracks which classifiers have been attempted per sequence
4. **Global Consistency**: Verifies working files match global metadata

### Error Handling

1. **Request Failures**: Retry failed requests up to max_retries
2. **Invalid Responses**: Log and continue with other classifiers
3. **File Corruption**: Detect and repair corrupted working files
4. **Process Interruption**: Safe to kill and restart processes

## Integration with Existing Pipelines

### Upstream Dependencies

- **jsonl2vocab**: Provides corpus.jsonl input file
- **PretrainRecord schema**: Uses established schema for sequence format

### Downstream Usage

- **Training Pipeline**: annotations.jsonl feeds into span boundary training
- **Evaluation Pipeline**: Working files enable annotation quality analysis
- **Factorized Pointer Network**: Direct input for supervised training

### Schema Compatibility

All output formats maintain compatibility with existing X-Spanformer schemas:
- Uses PretrainRecord format for sequence metadata
- Generates training records compatible with span boundary training
- Maintains character-level indexing for alignment with embedding pipeline

This pipeline serves as the critical bridge between corpus generation (jsonl2vocab) and supervised training, providing the hierarchical span annotations needed for the factorized pointer network training described in Section 3.3 of the X-Spanformer paper.
