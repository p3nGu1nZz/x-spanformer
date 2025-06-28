# 🧾 `csv2jsonl.py` — SelfCrit Ingestion CLI

This script parses one or more CSV files containing extracted text spans and converts them to structured `.jsonl` records annotated with selfcrit judgments, filtering tags, and metadata.

### 🚀 Basic Usage

```bash
python csv2jsonl.py -i ./data/input.csv -o ./out/
```

This will:
- Read text spans from the `text` column of `input.csv`
- Score and filter each span using the `selfcrit.yaml` config
- Output `dataset.jsonl` and optionally `dataset.json` to `./out/`

---

### ⚙️ CLI Options

| Flag            | Description                                                | Example                                      |
|-----------------|------------------------------------------------------------|----------------------------------------------|
| `-i`, `--input`  | Path to a CSV file or folder of `.csv` files               | `data/spans.csv` or `data/`                  |
| `-o`, `--output` | Output directory (created if missing)                      | `out/`                                       |
| `-f`, `--field`  | Column name to extract text spans from (default: `text`)   | `raw_text`                                   |
| `--pretty`       | Also write `.json` with pretty-formatted indentation       | `--pretty`                                   |
| `-n`, `--name`   | Base output filename (no extension)                        | `xspan-segments`                             |
| `--workers`      | Number of concurrent critique jobs (default: `1`)          | `--workers 8`                                |

---

### 📤 Output

- `dataset.jsonl`: newline-delimited JSON records (`PretrainRecord`)
- `dataset.json`: optional pretty version for inspection
- Each record is annotated with:
  - `score`, `status`, and `reason` from selfcrit
  - Tags like `revise`, `discard` (or empty if accepted)
  - Regex-triggered discards bypass model entirely

---

### 🖼 Sample Record

```json
{
  "raw": "The mitochondria is the powerhouse of the cell.",
  "meta": {
    "source_file": "biology-text.csv",
    "doc_language": "en",
    "extracted_by": "pdf2seg (manifest v1)",
    "confidence": 0.94,
    "tags": ["revised"],
    "notes": "clear structure, fluent segment"
  }
}
```

---

### 📡 Notes

- SelfCrit config is defined in `agents/config/selfcrit.yaml`
- Templates rendered from `agents/templates/`
- Regex-based noise filters are applied before any model call
