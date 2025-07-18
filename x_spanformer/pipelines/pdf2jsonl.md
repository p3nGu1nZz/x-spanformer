# 🧾 `pdf2jsonl.py` — PDF to JSONL Judge Ingestion CLI

This pipeline processes PDF files by first converting them to CSV using the `pdf2seg` Python package, then running Judge evaluation on the extracted text segments, and finally outputting structured JSONL records compatible with the X-Spanformer schema.

## Overview

The pipeline follows this workflow:
1. **PDF Discovery**: Find PDF files in the input directory (or process a single PDF file)
2. **Text Extraction**: Use `pdf2seg` package to extract text segments into CSV format
3. **Ollama Connection Test**: Test connection to Ollama with retry logic (3 attempts max)
4. **Resume Check**: Load existing dataset records to skip already processed segments
5. **Judge Evaluation**: Process each text segment through sequential Judge evaluation using configured LLM
6. **JSONL Generation**: Output structured records using `PretrainRecord` schema with metadata
7. **Incremental Saving**: Support progressive saving to prevent data loss during processing

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
| `--workers`      | Number of concurrent judge evaluations (default: `4`)          | `--workers 8`                                |
| `--save-interval`| Save dataset incrementally after every N segments          | `--save-interval 10`                        |
| `--force`        | Force regeneration of all cached data                      | `--force`                                    |

---

### 📤 Output

- `dataset.jsonl`: newline-delimited JSON records (`PretrainRecord`)
- `dataset.json`: optional pretty version for inspection  
- Each record contains:
  - `raw`: The original text segment from PDF
  - `type`: Content type classification (Natural, Code, Mixed)
  - `id`: Globally unique record identifier (auto-generated)
  - `meta`: Metadata object with the following fields:
    - `status`: Processing status (`"keep"`, `"discard"`)
    - `tags`: List of strings (mirrors status for non-keep records)
    - `doc_language`: ISO language code (e.g., `"en"`, `"ja"`)
    - `extracted_by`: Tool identifier (e.g., `"pdf2seg"`)
    - `confidence`: Float score from Judge evaluation (0.0-1.0)
    - `source_file`: Original PDF filename
    - `notes`: Judge reasoning/explanation

---

### 🖼 Sample Record

```json
{
  "raw": "The mitochondria is the powerhouse of the cell.",
  "type": "Natural",
  "id": {"id": "3d3e1e3e-8f6b-4a9a-9fc6-efedc5f805a8"},
  "meta": {
    "status": "keep",
    "tags": [],
    "doc_language": "en", 
    "extracted_by": "pdf2seg",
    "confidence": 0.94,
    "source_file": "biology-text.pdf",
    "notes": "clear structure, fluent segment"
  }
}
```

---

### 🔄 Workflow Details

1. **PDF Processing**: Uses `pdf2seg` package to convert PDFs to structured CSV with text spans
2. **Connection Testing**: Tests Ollama connection with retry logic before processing (max 3 attempts)
3. **Resume Support**: Automatically detects existing dataset files and skips already processed segments
4. **Text Splitting**: Automatically splits long text segments to fit model context limits
5. **Sequential Judge Evaluation**: Each text segment is evaluated by multiple LLM judges sequentially for better stability
6. **Retry Logic**: Failed judge evaluations are retried up to 3 times before failing
7. **Schema Compliance**: All records follow `PretrainRecord` schema with proper metadata
8. **Language Detection**: Automatic language detection using `langid` package
9. **Incremental Saving**: Prevents data loss during long processing runs
10. **Error Handling**: Program exits immediately when max retries are exhausted

### 🔄 Resume Functionality

The pipeline now supports **automatic resume** from interruptions:

- **Detection**: Automatically finds existing `dataset.jsonl` files in the output directory
- **Loading**: Reads all existing records and creates a skip list based on processed text segments
- **Filtering**: Only processes new segments that haven't been evaluated yet
- **Seamless**: No configuration needed - resume happens automatically

This means you can safely cancel a long-running job and restart it later without losing progress or duplicating work.

---

### 📡 Dependencies

- **pdf2seg**: PDF text extraction and segmentation (install separately)
- **Judge Agent**: LLM-based evaluation system (configured via `agents/config/judge.yaml`)
- **Rich Console**: Progress display and formatting
- **Ollama**: LLM backend for Judge evaluation

### 📡 Notes

- Judge config is defined in `agents/config/judge.yaml`
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
| `--workers` | | int | `1` | Number of concurrent workers for Judge evaluation |
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
    "tags": ["keep", "discard"],
    "doc_language": "en",
    "extracted_by": "pdf2seg",
    "confidence": 0.85,
    "source_file": "document.pdf",
    "notes": "Judge evaluation reason"
  }
}
```

### Metadata Fields

- **`tags`**: List of strings indicating Judge status (`["keep"]`, `["discard"]`, or `[]` for keep)
- **`doc_language`**: ISO language code detected by `langid`
- **`extracted_by`**: Tool information from PDF manifest or "unknown"
- **`confidence`**: Judge score (0.0-1.0)
- **`source_file`**: Original PDF filename
- **`notes`**: Judge reasoning for the evaluation decision

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
- Configured Judge LLM service (Ollama recommended)
- Sufficient disk space for temporary CSV files

## Configuration

The pipeline uses the Judge configuration system. Ensure you have:

1. **Judge Config**: Properly configured LLM model, temperature, and evaluation parameters
2. **LLM Service**: Running Ollama or compatible service
3. **Templates**: Judge evaluation templates loaded

Example configuration output:
```
═══ Judge Configuration ═══
Model: llama3.2:3b @ T=0.7
Voting: 3 passes | Retry: 2
Regex filters: 5
Templates: segment_judge
```

## Error Handling

The pipeline includes robust error handling for:

- **Missing pdf2seg**: Graceful fallback when package is not installed
- **PDF Processing Errors**: Continues with remaining files if individual PDFs fail
- **Ollama Connection Failures**: Tests connection with retry logic (3 attempts max) before starting
- **Judge Failures**: Retries individual judge evaluations up to 3 times before failing
- **Max Retry Exhaustion**: Exits program immediately when retries are exhausted
- **Network Issues**: Sequential judge processing reduces queue pressure on Ollama
- **File I/O Errors**: Validates CSV generation and JSONL writing

## Performance

- **Sequential Judge Processing**: Reduces Ollama queue pressure for better stability
- **Retry Logic**: Individual judge calls are retried up to 3 times before failing
- **Connection Testing**: Tests Ollama connectivity before starting processing
- **Incremental Saving**: Prevents data loss during long processing runs
- **Concurrent Evaluation**: Configurable worker count for parallel document processing
- **Memory Efficient**: Streams processing of large PDF collections
- **Progress Tracking**: Rich console output with detailed progress and statistics
- **Automatic Exit**: Program exits immediately when max retries are exhausted

## Integration with X-Spanformer

The generated JSONL files are compatible with:
- **Schema validation**: Uses `PretrainRecord` and `RecordMeta` models
- **Training pipelines**: Direct input for span-aware pretraining
- **Quality filtering**: Judge tags enable downstream filtering
- **Metadata tracking**: Full provenance from PDF source to processed record
