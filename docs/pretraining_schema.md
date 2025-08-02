# X-Spanformer Supervised Training Schema

**Specification for Agentic X-bar Span Annotation Pipeline**

This document defines the annotation pipeline and data format for generating supervised training data for the factorized pointer network span predictor. The pipeline uses multi-turn agentic conversations with phi4-mini to generate hierarchical X-bar span labels from raw Unicode sequences in our corpus data.

## Pipeline Overview

The annotation pipeline processes raw Unicode sequences from our corpus.jsonl (5,107 filtered sequences, 64-512 characters) through a multi-turn conversation system to generate X-bar span boundary annotations. These annotations feed directly into the masking-based training process for the factorized pointer network boundary prediction heads.

### Key Design Principles

1. **Single-turn Independent Requests**: Each hierarchical level gets its own independent request with domain-specific prompts
2. **Single-label Records**: Each training record contains exactly one span annotation for boundary masking training
3. **Hierarchical Spans**: Generate overlapping spans at word → phrase → clause → sentence levels (no length restrictions)
4. **Domain-aware Prompting**: Use tailored prompt templates for natural/code/mixed content types
5. **Resumable Processing**: Failed requests don't block other levels; can restart from any point
6. **Quality Focus**: Skip malformed responses, prioritize valid annotations over complete coverage

## Annotation Architecture

The annotation system processes each sequence comprehensively through systematic xbar.py classifier extraction. For detailed implementation specifications, see the `span_annotator.md` pipeline documentation.

### Comprehensive Processing Strategy

Each sequence undergoes complete processing for all applicable classifiers from `xbar.py`:

1. **Domain Detection**: Automatic identification from corpus sequence type (natural/code/mixed)
2. **Classifier Selection**: All applicable xbar.py classifiers selected based on domain
3. **Systematic Extraction**: Each classifier attempted using domain-specific prompts
4. **Statistical Tracking**: Success/failure rates tracked per classifier type
5. **Quality Focus**: Invalid responses logged but don't block other classifiers

### Agent Pattern Integration

The annotation system follows established patterns from the `agents/` package:

- **Jinja Templates**: Domain-specific templates for each xbar.py classifier type
- **OllamaClient**: Direct integration with existing model client infrastructure  
- **Dialogue Management**: Consistent request/response handling
- **Configuration**: YAML-based configuration following pipeline conventions

### Template Organization

Templates are systematically organized by domain and classifier type:

```
templates/{domain}/{classifier_type}/{classifier}.j2
```

Examples:
- `natural/labels/noun.j2` - Extract noun spans from natural language
- `code/roles/function.j2` - Extract function constructions from code
- `mixed/roles/inline_code.j2` - Extract inline code from mixed content
- `xbar/roles/head.j2` - Extract head elements (universal)

### Processing Architecture

The pipeline supports scalable annotation through:

- **Range-based Processing**: Specify sequences via `--range 1-100` for parallel coordination
- **Individual Working Files**: Each sequence gets its own JSON file for inspection
- **Statistical Tracking**: Comprehensive success/failure metrics per classifier
- **Resumable Processing**: Continue from any point, skip completed classifiers
- **Multi-process Support**: Multiple processes handle different sequence ranges


## Training Data Format

### Single-Label Record Structure

Each training record contains exactly one span annotation with Unicode character boundaries, enabling direct alignment with the masking-based factorized pointer network training objective from Section 3.3 of the paper.

```json
{
  "corpus_id": "corpus-seq-00000001",
  "sequence_number": 1,
  "raw_sequence": "The quick brown fox jumps over the lazy dog.",
  "char_start": 4,
  "char_end": 18,
  "span_text": "quick brown fox",
  "xbar_label": "noun_phrase",
  "xbar_role": "subject",
  "hierarchical_level": "phrase",
  "domain_type": "natural",
  "source_meta": {
    "status": "keep",
    "extracted_by": "jsonl2vocab", 
    "timestamp": "2025-07-25T22:34:12.396758",
    "source": "consolidated_corpus"
  }
}
```

### Field Definitions

- `corpus_id`: Sequence identifier from corpus.jsonl id.id field (e.g., "corpus-seq-00000001")
- `sequence_number`: Sequential number from corpus.jsonl meta.sequence_number for ordering
- `raw_sequence`: Complete Unicode string from PretrainRecord.raw field
- `char_start`: Start character position in raw_sequence (inclusive, 0-indexed)
- `char_end`: End character position in raw_sequence (inclusive, 0-indexed)  
- `span_text`: Substring extracted from raw_sequence[char_start:char_end+1]
- `xbar_label`: X-bar category from simplified taxonomy
- `xbar_role`: Functional role within syntactic structure
- `hierarchical_level`: One of ["word", "phrase", "clause", "sentence"]
- `domain_type`: One of ["natural", "code", "mixed"] from PretrainRecord.type
- `source_meta`: Original metadata from corpus.jsonl for traceability

**Position-wise Embedding Alignment**: In X-Spanformer's tokenizer-free architecture, each Unicode character corresponds directly to one embedding position. Therefore, `char_start` and `char_end` map directly to position indices used in the span predictor training (Section 3.3 of the paper).

**Important**: Each position embedding H[t] ∈ R^512 represents the contextual character representation at position t, NOT a span embedding. Spans are detected through boundary prediction:
- Start boundary: `y_start[char_start] = 1.0`
- End boundary: `y_end[char_end] = 1.0`

The factorized pointer network learns to predict these boundaries from contextual position embeddings, enabling detection of overlapping and variable-length spans.

## Data Storage & Training Integration

### Storage Architecture

The annotation pipeline uses individual working files for scalable processing and easy inspection. For detailed storage specifications, see `span_annotator.md` pipeline documentation.

**Key Features:**
- Individual JSON files per sequence for parallel processing
- Global metadata for progress tracking and statistics
- Consolidated JSONL output for training
- Resume capability from any point

### Training Integration

The annotations feed directly into the factorized pointer network training process:

1. **Span Masking**: Hide target spans during training
2. **Boundary Prediction**: Train start/end position heads with sigmoid + BCE loss
3. **Position-wise Alignment**: Character indices map directly to embedding positions (1:1 correspondence)
4. **Multi-label Support**: BCE loss handles overlapping spans

```python
# Training flow example with position-wise alignment
raw_seq = record["raw_sequence"]
char_start, char_end = record["char_start"], record["char_end"]

# In tokenizer-free architecture: char_position = embedding_position
pos_start, pos_end = char_start, char_end

# Mask span and generate contextual embeddings
masked_seq = mask_span(raw_seq, pos_start, pos_end)
H = embeddings_pipeline(masked_seq)  # From vocab2embedding (Section 3.2)

# Binary boundary targets for factorized pointer network (Section 3.3)
y_start[pos_start] = 1.0  
y_end[pos_end] = 1.0

# Train boundary heads
logits_start = linear_start(H)
logits_end = linear_end(H)
loss = bce_loss(sigmoid(logits_start), y_start) + bce_loss(sigmoid(logits_end), y_end)
```

## Simplified X-bar Taxonomy

To ensure practical annotation success, we use a simplified subset of the full X-bar taxonomy defined in `xbar.py`. This "acid test" taxonomy focuses on the most frequent and reliably annotatable categories.

### Natural Language Labels

**Word Level:**
- `noun`, `verb`, `adjective`, `adverb`, `determiner`, `preposition`, `pronoun`, `conjunction`, `punctuation`

**Phrase Level:**
- `noun_phrase`, `verb_phrase`, `adjective_phrase`, `adverb_phrase`, `prepositional_phrase`

**Clause Level:**
- `main_clause`, `subordinate_clause`, `relative_clause`

**Sentence Level:**
- `simple_sentence`, `compound_sentence`, `complex_sentence`

### Code Labels

**Word Level:**
- `keyword`, `identifier`, `operator`, `literal`, `delimiter`, `type_name`, `comment`

**Phrase Level:**
- `expression`, `function_call`, `assignment`, `parameter_list`, `argument_list`

**Clause Level:**
- `if_statement`, `loop_statement`, `function_definition`, `class_definition`

**Sentence Level:**
- `statement_block`, `function_body`, `class_body`, `module`

### Mixed Domain Labels

**Transition Markers:**
- `code_inline`, `code_block`, `natural_instruction`, `natural_explanation`

**Hybrid Constructs:**
- `documented_code`, `instructional_sequence`, `example_block`

## Training Data Generation Pipeline

### Command Line Interface

```bash
# Single sequence annotation - extracts all possible xbar.py classifiers
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1

# Multiple sequences annotation
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1,5,10

# Range of sequences annotation  
python -m x_spanformer.pipelines.span_annotator \
  --corpus data/vocab/corpus.jsonl \
  --output data/annotations \
  --range 1-100
```

### Processing Strategy

The pipeline processes sequences comprehensively, attempting to extract all possible xbar.py classifiers from each sequence:

1. **Range Selection**: Use `--range` to specify sequences to process:
   - Single sequence: `--range 1`
   - Multiple sequences: `--range 1,5,10`
   - Range of sequences: `--range 1-100`
2. **Comprehensive Annotation**: Each sequence is processed for ALL xbar.py classifiers:
   - Automatically detects domain type from corpus.jsonl (natural/code/mixed)
   - Attempts extraction for all applicable label types based on domain
   - Uses domain-appropriate Jinja templates for each classifier
   - Tracks success/failure for each classifier attempt in working files
3. **Auto-managed Structure**: Output directory automatically creates:
   - `working/` for individual sequence JSON files (corpus-seq-XXXXXXXX.json)
   - `consolidated/` for final training JSONL
   - `metadata.json` for progress tracking
4. **Parallel Processing**: Multiple processes can run simultaneously on different sequence ranges

### Multi-Process Architecture

Multiple span annotator processes can run simultaneously on different sequence ranges to achieve parallelism:

```python
def process_sequence_range(corpus_file: Path, output_dir: Path, sequence_range: str):
    """Process a specific range of sequences with comprehensive xbar.py classifier extraction."""
    
    # Parse range specification
    if '-' in sequence_range:
        start, end = map(int, sequence_range.split('-'))
        sequence_ids = list(range(start, end + 1))
    elif ',' in sequence_range:
        sequence_ids = list(map(int, sequence_range.split(',')))
    else:
        sequence_ids = [int(sequence_range)]
    
    # Load corpus and filter to target sequences
    target_sequences = []
    with open(corpus_file) as f:
        for line_num, line in enumerate(f, 1):
            if line_num in sequence_ids:
                record = json.loads(line)
                target_sequences.append(record)
    
    # Process each sequence with comprehensive classifier extraction
    session = SpanAnnotatorSession()
    for record in target_sequences:
        # Extract all applicable xbar.py classifiers based on domain
        domain_type = record['type']  # natural, code, or mixed
        
        classifier_results = session.extract_all_classifiers(
            raw_sequence=record['raw'],
            corpus_id=record['id']['id'], 
            domain_type=domain_type
        )
        
        # Save comprehensive results to individual working file
        working_file = output_dir / "working" / f"{record['id']['id']}.json"
        save_comprehensive_results(working_file, record, classifier_results)
        
        # Update global metadata with classifier statistics
        update_classifier_metadata(output_dir / "metadata.json", classifier_results)

class SpanAnnotatorSession:
    """Comprehensive xbar.py classifier extraction session."""
    
    def __init__(self, config_file="span_annotator.yaml"):
        self.cfg = load_config(config_file)
        self.ollama_client = AsyncClient()
        self.jinja_env = setup_jinja_templates()
        
    def extract_all_classifiers(self, raw_sequence: str, corpus_id: str, domain_type: str) -> dict:
        """Extract all applicable xbar.py classifiers from a sequence."""
        results = {
            "sequence_info": {"corpus_id": corpus_id, "domain_type": domain_type},
            "classifier_attempts": {},
            "successful_extractions": [],
            "failed_extractions": []
        }
        
        # Get applicable classifiers based on domain
        classifiers = self.get_applicable_classifiers(domain_type)
        
        for classifier_type, classifier_list in classifiers.items():
            results["classifier_attempts"][classifier_type] = {}
            
            for classifier in classifier_list:
                try:
                    # Use domain-specific Jinja template
                    template_path = f"{domain_type}/{classifier_type.lower()}/{classifier}.j2"
                    spans = await self.extract_classifier_spans(
                        raw_sequence, template_path, classifier
                    )
                    
                    if spans:
                        results["classifier_attempts"][classifier_type][classifier] = {
                            "status": "success",
                            "count": len(spans),
                            "timestamp": datetime.now().isoformat()
                        }
                        results["successful_extractions"].extend(spans)
                    else:
                        results["classifier_attempts"][classifier_type][classifier] = {
                            "status": "failed",
                            "error": "no matches found",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    results["classifier_attempts"][classifier_type][classifier] = {
                        "status": "failed", 
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    results["failed_extractions"].append({
                        "classifier": classifier,
                        "classifier_type": classifier_type,
                        "error": str(e)
                    })
        
        return results
    
    def get_applicable_classifiers(self, domain_type: str) -> dict:
        """Get applicable xbar.py classifiers based on domain type."""
        base_classifiers = {
            "XBarRole": ["specifier", "complement", "adjunct", "head", "modifier", "determiner", "nucleus"]
        }
        
        if domain_type == "natural":
            return {
                **base_classifiers,
                "NaturalLabel": ["noun", "verb", "adjective", "adverb", "preposition", "conjunction", "determiner", "pronoun", "interjection", "punctuation"],
                "NaturalRole": ["subject", "object", "complement", "predicate", "modifier", "attributive", "predicative"]
            }
        elif domain_type == "code":
            return {
                **base_classifiers,
                "CodeLabel": ["keyword", "identifier", "operator", "delimiter", "literal", "type", "comment"],
                "CodeRole": ["function", "declaration", "expression", "statement", "parameter", "argument"]
            }
        elif domain_type == "mixed":
            return {
                **base_classifiers,
                "NaturalLabel": ["noun", "verb", "adjective", "identifier"],  # Subset for mixed
                "CodeLabel": ["keyword", "identifier", "operator"],  # Subset for mixed  
                "HybridRole": ["inline_code", "documentation_context", "transition_marker"]
            }
        
        return base_classifiers

# Example: Run multiple processes for different sequence ranges
ranges = ["1-100", "101-200", "201-300", "301-400", "401-500"]

processes = []
for range_spec in ranges:
    p = subprocess.Popen([
        "python", "-m", "x_spanformer.pipelines.span_annotator",
        "--corpus", "data/vocab/corpus.jsonl",
        "--output", "data/annotations", 
        "--range", range_spec
    ])
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.wait()
```

### Output Format: annotations.jsonl

Each line contains a single span annotation record with Unicode character boundaries and full corpus traceability:

```jsonl
{"corpus_id": "corpus-seq-00000001", "sequence_number": 1, "raw_sequence": "The cat sleeps.", "char_start": 0, "char_end": 2, "span_text": "The", "xbar_label": "determiner", "xbar_role": "specifier", "hierarchical_level": "word", "domain_type": "natural", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000001", "sequence_number": 1, "raw_sequence": "The cat sleeps.", "char_start": 4, "char_end": 6, "span_text": "cat", "xbar_label": "noun", "xbar_role": "head", "hierarchical_level": "word", "domain_type": "natural", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000001", "sequence_number": 1, "raw_sequence": "The cat sleeps.", "char_start": 0, "char_end": 6, "span_text": "The cat", "xbar_label": "noun_phrase", "xbar_role": "subject", "hierarchical_level": "phrase", "domain_type": "natural", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
```

### Training Dataset Statistics

For initial validation of our theoretical approach, we target:

- **Minimum viable dataset**: ~10,000 annotated spans across all levels
- **Hierarchical distribution**: 40% word, 30% phrase, 20% clause, 10% sentence
- **Domain balance**: 50% natural, 30% code, 20% mixed
- **Quality threshold**: >80% candidate validation rate where applicable

This scale provides sufficient signal to validate:
1. Sigmoid boundary prediction effectiveness
2. Binary cross-entropy training convergence  
3. Factorized pointer network span scoring
4. Multi-label overlapping span handling

## Example Annotations by Domain

### Natural Language Example

**Source**: "The quick brown fox jumps over the lazy dog." (corpus-seq-00000001)

**Generated Records**:
```jsonl
{"corpus_id": "corpus-seq-00000001", "sequence_number": 1, "raw_sequence": "The quick brown fox jumps over the lazy dog.", "char_start": 0, "char_end": 2, "span_text": "The", "xbar_label": "determiner", "xbar_role": "specifier", "hierarchical_level": "word", "domain_type": "natural", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000001", "sequence_number": 1, "raw_sequence": "The quick brown fox jumps over the lazy dog.", "char_start": 4, "char_end": 18, "span_text": "quick brown fox", "xbar_label": "noun_phrase", "xbar_role": "subject", "hierarchical_level": "phrase", "domain_type": "natural", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000001", "sequence_number": 1, "raw_sequence": "The quick brown fox jumps over the lazy dog.", "char_start": 0, "char_end": 43, "span_text": "The quick brown fox jumps over the lazy dog", "xbar_label": "simple_sentence", "xbar_role": "complete_thought", "hierarchical_level": "sentence", "domain_type": "natural", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
```

### Code Example

**Source**: "let x = 42;" (corpus-seq-00000002)

**Generated Records**:
```jsonl
{"corpus_id": "corpus-seq-00000002", "sequence_number": 2, "raw_sequence": "let x = 42;", "char_start": 0, "char_end": 2, "span_text": "let", "xbar_label": "keyword", "xbar_role": "declaration", "hierarchical_level": "word", "domain_type": "code", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000002", "sequence_number": 2, "raw_sequence": "let x = 42;", "char_start": 4, "char_end": 9, "span_text": "x = 42", "xbar_label": "assignment", "xbar_role": "initialization", "hierarchical_level": "phrase", "domain_type": "code", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000002", "sequence_number": 2, "raw_sequence": "let x = 42;", "char_start": 0, "char_end": 10, "span_text": "let x = 42;", "xbar_label": "statement_block", "xbar_role": "variable_declaration", "hierarchical_level": "sentence", "domain_type": "code", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
```

### Mixed Domain Example

**Source**: "To define a constant, use `const PI = 3.14`." (corpus-seq-00000003)

**Generated Records**:
```jsonl
{"corpus_id": "corpus-seq-00000003", "sequence_number": 3, "raw_sequence": "To define a constant, use `const PI = 3.14`.", "char_start": 26, "char_end": 42, "span_text": "`const PI = 3.14`", "xbar_label": "code_inline", "xbar_role": "example", "hierarchical_level": "phrase", "domain_type": "mixed", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
{"corpus_id": "corpus-seq-00000003", "sequence_number": 3, "raw_sequence": "To define a constant, use `const PI = 3.14`.", "char_start": 0, "char_end": 19, "span_text": "To define a constant", "xbar_label": "natural_instruction", "xbar_role": "goal", "hierarchical_level": "phrase", "domain_type": "mixed", "source_meta": {"status": "keep", "extracted_by": "jsonl2vocab", "timestamp": "2025-07-25T22:34:12.396758", "source": "consolidated_corpus"}}
```

## Integration with Span Predictor Training

The generated `annotations.jsonl` directly feeds into the masking-based supervised training process described in Section 3.3 of the paper:

1. **Masking Strategy**: Hide target spans and train boundary prediction on raw Unicode sequences
2. **Position-wise Boundary Alignment**: Character indices map directly to embedding positions (1:1 correspondence)
3. **Multi-Label Support**: Multiple records per sequence create overlapping boundary labels
4. **Hierarchical Training**: Different levels enable progressive training complexity
5. **No Length Restrictions**: Sentence/clause spans can exceed 84-character w_max limit

### Architecture Flow Integration

The annotation format aligns with the complete span predictor architecture:

```python
# Load annotations and create training batch
raw_sequences = [record["raw_sequence"] for record in batch]
span_boundaries = [(record["char_start"], record["char_end"]) for record in batch]

# Step 1: Generate seed embeddings (Section 3.2)
seed_embeddings = vocab_to_embedding(raw_sequences)

# Step 2: Create contextual embeddings H ∈ R^(T×d) (Section 3.2)  
H = seed_to_contextual(seed_embeddings)

# Step 3: Factorized pointer network boundary prediction (Section 3.3)
logits_start = H @ W_start + b_start  # Linear projection heads
logits_end = H @ W_end + b_end
probs_start = sigmoid(logits_start)   # Independent boundary probabilities  
probs_end = sigmoid(logits_end)

# Step 4: Create binary targets from position boundaries (char_pos = embedding_pos)
y_start, y_end = create_boundary_targets(span_boundaries, sequence_lengths)

# Step 5: Multi-label binary cross-entropy loss (supports overlapping spans)
loss = bce_loss(probs_start, y_start) + bce_loss(probs_end, y_end)
```

## Implementation Roadmap

### Phase 1: Core Pipeline (Current)
- [ ] Implement `SpanAnnotatorSession` class with comprehensive xbar.py classifier extraction
- [ ] Create Jinja template system organized by domain/classifier_type/classifier.j2 structure  
- [ ] Build range-based sequence processing with complete classifier coverage
- [ ] Implement working file management with classifier attempt tracking

### Phase 2: Template Development  
- [ ] Develop Jinja templates for all xbar.py classifier types (50+ templates)
- [ ] Implement domain-aware classifier selection (natural/code/mixed → applicable classifiers)
- [ ] Add Ollama client integration following judge_session.py pattern
- [ ] Create JSON response parsing and validation for each classifier type

### Phase 3: Comprehensive Processing
- [ ] Build multi-process coordination for sequence range processing
- [ ] Implement classifier success/failure tracking with statistics
- [ ] Add progress monitoring with classifier-level granularity  
- [ ] Create distribution analysis to optimize annotation efficiency

### Phase 4: Training Integration
- [ ] Consolidation pipeline from comprehensive working files → training JSONL
- [ ] Statistical analysis of classifier success rates and span distributions
- [ ] Integration with factorized pointer network training pipeline
- [ ] Quality metrics for overall annotation coverage and consistency

### Comprehensive X-Bar Annotation Architecture

The simplified CLI enables comprehensive processing where each sequence is exhaustively processed for all applicable xbar.py classifiers:

```python
class ComprehensiveXBarAnnotator:
    """Comprehensive xbar.py classifier extraction for sequence-level annotation."""
    
    def __init__(self, config_file="span_annotator.yaml"):
        self.cfg = load_config(config_file)
        self.ollama_client = AsyncClient()
        self.templates = self._load_classifier_templates()
        
    def process_sequence(self, sequence: dict) -> dict:
        """Process one sequence for all applicable xbar.py classifiers."""
        domain_type = sequence['type']
        raw_text = sequence['raw']
        corpus_id = sequence['id']['id']
        
        # Get all applicable classifiers for this domain
        applicable_classifiers = self.get_classifiers_for_domain(domain_type)
        
        results = {
            "corpus_id": corpus_id,
            "domain_type": domain_type,
            "classifier_attempts": {},
            "annotations": [],
            "failed_requests": []
        }
        
        # Attempt extraction for each applicable classifier
        for classifier_type, classifiers in applicable_classifiers.items():
            results["classifier_attempts"][classifier_type] = {}
            
            for classifier in classifiers:
                template_path = f"{domain_type}/{classifier_type.lower()}/{classifier}.j2"
                
                try:
                    spans = await self.extract_spans_with_template(
                        raw_text, template_path, classifier, classifier_type
                    )
                    
                    if spans:
                        results["classifier_attempts"][classifier_type][classifier] = {
                            "status": "success",
                            "count": len(spans),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Add to annotations list
                        for span in spans:
                            results["annotations"].append({
                                **span,
                                "xbar_label": classifier,
                                "classifier_type": classifier_type,
                                "domain_type": domain_type
                            })
                    else:
                        results["classifier_attempts"][classifier_type][classifier] = {
                            "status": "failed",
                            "error": "no matches found",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                except Exception as e:
                    results["classifier_attempts"][classifier_type][classifier] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    results["failed_requests"].append({
                        "classifier": classifier,
                        "classifier_type": classifier_type,
                        "error": str(e)
                    })
        
        return results
    
    def get_classifiers_for_domain(self, domain_type: str) -> dict:
        """Return applicable xbar.py classifiers based on domain type."""
        # Always include XBarRole classifiers
        base = {"XBarRole": list(XBarRole.__args__)}
        
        if domain_type == "natural":
            return {
                **base,
                "NaturalLabel": list(NaturalLabel.__args__),
                "NaturalRole": list(NaturalRole.__args__)
            }
        elif domain_type == "code":
            return {
                **base,
                "CodeLabel": list(CodeLabel.__args__),
                "CodeRole": list(CodeRole.__args__)
            }
        elif domain_type == "mixed":
            return {
                **base,
                "NaturalLabel": list(NaturalLabel.__args__)[:5],  # Subset
                "CodeLabel": list(CodeLabel.__args__)[:5],       # Subset
                "HybridRole": list(HybridRole.__args__)
            }
        
        return base
```

This approach provides:
- **Complete Coverage**: Every sequence processed for all applicable classifiers
- **Quality Statistics**: Success rates tracked per classifier type
- **Efficient Scaling**: 50k+ requests distributed across classifier attempts
- **Domain Awareness**: Automatic classifier selection based on sequence domain
- **Statistical Analysis**: Rich data for optimizing training data distribution
    
    def mark_completed(self, record_id: str, level: str, domain: str):
        """Mark this combination as completed."""
        key = f"{record_id}:{level}:{domain}"
        self.completed.add(key)
        self._save_progress()
    
    def get_pending_tasks(self, sequences: List[PretrainRecord], 
                         levels: List[str]) -> List[Tuple[str, str, str]]:
        """Get all (record_id, level, domain) combinations that need processing."""
        pending = []
        for record in sequences:
            for level in levels:
                if not self.is_completed(str(record.id.id), level, record.type):
                    pending.append((str(record.id.id), level, record.type))
        return pending
```

This architecture provides maximum flexibility for generating training data at scale while maintaining quality and enabling efficient resource utilization.
