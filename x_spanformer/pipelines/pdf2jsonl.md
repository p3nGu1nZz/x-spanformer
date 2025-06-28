# 🧾 `pdf2jsonl.py` — PDF to JSONL SelfCrit Ingestion CLI

This pipeline processes PDF files by first converting them to CSV using the `pdf2seg` Python package, then running SelfCrit evaluation on the extracted text segments, and finally outputting structured JSONL records compatible with the X-Spanformer schema.

## Overview

The pipeline follows this workflow:
1. **PDF Discovery**: Find PDF files in the input directory (or process a single PDF file)
2. **Text Extraction**: Use `pdf2seg` package to extract text segments into CSV format
3. **SelfCrit Evaluation**: Process each text segment through SelfCrit evaluation using configured LLM
4. **JSONL Generation**: Output structured records using `PretrainRecord` schema with metadata
5. **Incremental Saving**: Support progressive saving to prevent data loss during processing

## Usage

```bash
python pdf2jsonl.py -i ./data/input.pdf -o ./out/
python pdf2jsonl.py -i ./data/pdfs/ -o ./out/ --pretty --workers 4
```

### ⚙️ CLI Options

| Flag            | Description                                                | Example                                      |
|-----------------|------------------------------------------------------------|----------------------------------------------|
| `-i`, `--input`  | Path to a PDF file or folder of `.pdf` files               | `data/paper.pdf` or `data/pdfs/`            |
| `-o`, `--output` | Output directory (created if missing)                      | `out/`                                       |
| `-f`, `--field`  | Column name to extract text spans from (default: `text`)   | `raw_text`                                   |
| `--pretty`       | Also write `.json` with pretty-formatted indentation       | `--pretty`                                   |
| `-n`, `--name`   | Base output filename (no extension)                        | `xspan-segments`                             |
| `--workers`      | Number of concurrent critique jobs (default: `1`)          | `--workers 8`                                |
| `--save-interval`| Save dataset incrementally after every N segments          | `--save-interval 10`                        |

---

### 📤 Output

- `dataset.jsonl`: newline-delimited JSON records (`PretrainRecord`)
- `dataset.json`: optional pretty version for inspection  
- Each record contains:
  - `raw`: The original text segment from PDF
  - `id`: Globally unique record identifier (auto-generated)
  - `meta`: Metadata object with the following optional fields:
    - `tags`: List of strings (e.g., `["keep"]`, `["discard"]`, `["revise"]`)
    - `doc_language`: ISO language code (e.g., `"en"`, `"ja"`)
    - `extracted_by`: Tool identifier (e.g., `"pdf2seg v1.0"`)
    - `confidence`: Float score from SelfCrit evaluation (0.0-1.0)
    - `source_file`: Original PDF filename
    - `notes`: SelfCrit reasoning/explanation

---

### 🖼 Sample Record

```json
{
  "raw": "The mitochondria is the powerhouse of the cell.",
  "id": {"id": "3d3e1e3e-8f6b-4a9a-9fc6-efedc5f805a8"},
  "meta": {
    "tags": [],
    "doc_language": "en", 
    "extracted_by": "pdf2seg (manifest v1)",
    "confidence": 0.94,
    "source_file": "biology-text.pdf",
    "notes": "clear structure, fluent segment"
  }
}
```

---

### 🔄 Workflow Details

1. **PDF Processing**: Uses `pdf2seg` package to convert PDFs to structured CSV with text spans
2. **SelfCrit Evaluation**: Each text segment is evaluated by LLM for training suitability
3. **Schema Compliance**: All records follow `PretrainRecord` schema with proper metadata
4. **Language Detection**: Automatic language detection using `langid` package
5. **Incremental Saving**: Prevents data loss during long processing runs

### 📡 Dependencies

- **pdf2seg**: PDF text extraction and segmentation (install separately)
- **SelfCrit Agent**: LLM-based evaluation system (configured via `agents/config/selfcrit.yaml`)
- **Rich Console**: Progress display and formatting
- **Ollama**: LLM backend for SelfCrit evaluation

### 📡 Notes

- SelfCrit config is defined in `agents/config/selfcrit.yaml`
- Templates rendered from `agents/templates/`
- Regex-based noise filters are applied before any model call
- CSV files are temporarily stored in `{output}/temp_csv/` during processing
- Supports both single PDF processing and batch directory processing
```

### Multiple PDFs
```bash
python pdf2jsonl.py -i ./data/pdfs/ -o ./out/ --workers 2 --save-interval 5
```

### Custom field and output naming
```bash
python pdf2jsonl.py -i ./input.pdf -o ./output/ -f "content" -n "my_dataset" --pretty
```

## Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | Path | *required* | Input PDF file or directory containing PDF files |
| `--output` | `-o` | Path | *required* | Output directory for JSONL files |
| `--field` | `-f` | str | `"text"` | Field name to process from the generated CSV |
| `--pretty` | | flag | `False` | Generate pretty-printed JSON output alongside JSONL |
| `--name` | `-n` | str | `"dataset"` | Base name for output files |
| `--workers` | | int | `1` | Number of concurrent workers for SelfCrit evaluation |
| `--save-interval` | | int | `10` | Save dataset incrementally after every N segments (0 to disable) |

## Output Schema

The pipeline generates `PretrainRecord` objects with the following structure:

```json
{
  "raw": "The extracted text segment from the PDF",
  "id": {
    "id": "uuid4-generated-identifier"
  },
  "meta": {
    "tags": ["keep", "revise", "discard"],
    "doc_language": "en",
    "extracted_by": "pdf2seg",
    "confidence": 0.85,
    "source_file": "document.pdf",
    "notes": "SelfCrit evaluation reason"
  }
}
```

### Metadata Fields

- **`tags`**: List of strings indicating SelfCrit status (`["keep"]`, `["revise"]`, `["discard"]`, or `[]` for keep)
- **`doc_language`**: ISO language code detected by `langid`
- **`extracted_by`**: Tool information from PDF manifest or "unknown"
- **`confidence`**: SelfCrit score (0.0-1.0)
- **`source_file`**: Original PDF filename
- **`notes`**: SelfCrit reasoning for the evaluation decision

## Dependencies

### Required Packages
- `pdf2seg` - PDF text extraction and segmentation package
- `langid` - Language detection
- `rich` - Console output formatting
- `ollama` - LLM client (or compatible API)
- `pydantic` - Schema validation
- `asyncio` - Asynchronous processing

### System Requirements
- Python 3.8+
- Configured SelfCrit LLM service (Ollama recommended)
- Sufficient disk space for temporary CSV files

## Configuration

The pipeline uses the SelfCrit configuration system. Ensure you have:

1. **SelfCrit Config**: Properly configured LLM model, temperature, and evaluation parameters
2. **LLM Service**: Running Ollama or compatible service
3. **Templates**: SelfCrit evaluation templates loaded

Example configuration output:
```
═══ SelfCrit Configuration ═══
Model: llama3.2:3b @ T=0.7
Voting: 3 passes | Retry: 2
Regex filters: 5
Templates: segment_judge, quality_check
```

## Error Handling

The pipeline includes robust error handling for:

- **Missing pdf2seg**: Graceful fallback when package is not installed
- **PDF Processing Errors**: Continues with remaining files if individual PDFs fail
- **SelfCrit Failures**: Assigns default "revise" status with error note
- **Network Issues**: Retries LLM calls according to configuration
- **File I/O Errors**: Validates CSV generation and JSONL writing

## Performance

- **Incremental Saving**: Prevents data loss during long processing runs
- **Concurrent Evaluation**: Configurable worker count for SelfCrit processing
- **Memory Efficient**: Streams processing of large PDF collections
- **Progress Tracking**: Rich console output with detailed progress and statistics

## Integration with X-Spanformer

The generated JSONL files are compatible with:
- **Schema validation**: Uses `PretrainRecord` and `RecordMeta` models
- **Training pipelines**: Direct input for span-aware pretraining
- **Quality filtering**: SelfCrit tags enable downstream filtering
- **Metadata tracking**: Full provenance from PDF source to processed record
