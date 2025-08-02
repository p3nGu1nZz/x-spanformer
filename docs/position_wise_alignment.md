# Position-wise Embedding Alignment

## Overview

X-Spanformer uses a **tokenizer-free architecture** where each Unicode character corresponds directly to one embedding position. This creates a 1:1 mapping between character indices and embedding positions, enabling seamless alignment between span annotations and contextual embeddings.

## Character-to-Position Mapping

### Direct Correspondence

In X-Spanformer's architecture:
- **Character index `i`** = **Embedding position `i`**
- **Character span `[start_char, end_char]`** = **Position span `[start_pos, end_pos]`**

### Example

```python
text = "The quick brown fox"
#      0123456789012345678
#      T h e   q u i c k   b r o w n   f o x

# Character-level annotation
char_span = {
    "text": "quick brown fox",
    "char_start": 4,    # Character index of 'q'
    "char_end": 18,     # Character index of 'x' (inclusive)
}

# Position-wise embedding alignment
position_span = {
    "text": "quick brown fox", 
    "start_pos": 4,     # Embedding position 4
    "end_pos": 19,      # Embedding position 19 (exclusive for range)
}

# The span covers embedding positions [4, 5, 6, ..., 18]
embedding_positions = list(range(4, 19))  # 15 positions total
```

## Schema Integration

### PretrainRecord Enhancement

The `PretrainRecord` schema includes fields for embedding alignment:

```python
class PretrainRecord(BaseModel):
    raw: str = Field(..., description="The raw Unicode text sequence")
    # ... existing fields ...
    
    # New fields for position-wise alignment
    sequence_number: Optional[int] = Field(default=None, description="Sequential position in corpus")
    embedding_chunk_id: Optional[int] = Field(default=None, description="Chunk ID containing embeddings")
    embedding_positions: Optional[int] = Field(default=None, description="Number of positions (len(raw))")
```

### Position Mapping Utilities

The `PositionMapper` class handles conversions between character and position spans:

```python
from x_spanformer.agents.position_mapper import PositionMapper

mapper = PositionMapper("The quick brown fox")

# Convert character span to position span
char_span = CharacterSpan(start_char=4, end_char=18, xbar_class="NP")
pos_span = mapper.char_span_to_position_span(char_span)

# pos_span.start_pos == 4
# pos_span.end_pos == 19
# pos_span.positions == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
```

## Training Integration

### Span Predictor Training (Section 3.3)

The position-wise alignment enables direct training of the factorized pointer network:

```python
# Load annotation record
record = {
    "raw_sequence": "The quick brown fox",
    "char_start": 4,
    "char_end": 18,
    "xbar_label": "noun_phrase"
}

# Direct position mapping (no conversion needed)
pos_start, pos_end = record["char_start"], record["char_end"]

# Generate contextual embeddings H ∈ R^(T×d)
H = vocab2embedding_pipeline(record["raw_sequence"])  # Shape: (19, d)

# Create binary boundary targets
y_start = torch.zeros(len(record["raw_sequence"]))  # Shape: (19,)
y_end = torch.zeros(len(record["raw_sequence"]))    # Shape: (19,)

y_start[pos_start] = 1.0  # Position 4
y_end[pos_end] = 1.0      # Position 18

# Train boundary prediction heads
logits_start = linear_start(H)  # Shape: (19, 1)
logits_end = linear_end(H)      # Shape: (19, 1)

# Multi-label binary cross-entropy loss
loss = bce_loss(sigmoid(logits_start), y_start) + bce_loss(sigmoid(logits_end), y_end)
```

## Advantages

### 1. **Seamless Integration**
- No tokenization step required
- No vocabulary alignment issues
- Direct character-to-embedding mapping

### 2. **Consistent Indexing**
- Character annotations map directly to training targets
- No index conversion errors
- Simplified pipeline integration

### 3. **Flexible Span Boundaries**
- Support for any character-level span
- No subword boundary constraints
- Natural handling of multilingual text

### 4. **Training Efficiency**
- Direct supervision signal
- No alignment preprocessing
- Simplified loss computation

## Paper Alignment

This approach aligns with Section 3.2 of the X-Spanformer paper:

> "The X-Spanformer pipeline begins with a sequence x = [x₁, x₂, ..., xₜ] of T raw Unicode codepoints... we compute position-wise soft piece probabilities and transform them into contextual embeddings"

The position-wise embeddings H ∈ R^(T×d) have exactly T positions corresponding to T Unicode characters, enabling the direct character-to-position mapping used throughout the annotation and training pipeline.
