# 🐙 `repo2jsonl.py` — GitHub Repository Export Pipeline

## Overview

The `repo2jsonl` pipeline exports GitHub repositories using shallow git clone and processes the source code files for training data generation. It uses `git clone --depth 1` to efficiently download only the latest commit, then removes the `.git` directory to avoid conflicts with the parent repository.

## Key Features

✅ **No API Keys Required** - Uses standard git clone (no GitHub API)  
✅ **Shallow Clone** - Only downloads latest commit (`--depth 1`)  
✅ **Clean Export** - Removes `.git` directory to avoid conflicts  
✅ **Smart Filtering** - Skips binary files, build artifacts, and data directories  
✅ **Configuration-Driven** - All settings loaded from `config/pipelines/repo2jsonl.yaml`  
✅ **Resume Support** - Automatically resumes from previous exports  
✅ **Branch Selection** - Can target specific branches  

## Installation

Install GitPython dependency:

```bash
pip install gitpython
```

## Configuration

The pipeline is fully configuration-driven using `config/pipelines/repo2jsonl.yaml`:

```yaml
# Repository export settings
repository:
  clone_depth: 1          # Shallow clone depth
  single_branch: true     # Only clone target branch
  remove_git_dir: true    # Remove .git after clone
  max_file_size: 50000    # Max file size in characters
  
  extensions:             # File extensions to include
    - .py                 # Python
    - .js                 # JavaScript
    # ... (20+ programming languages)
  
  skip_directories:       # Directories to skip
    - .git
    - data                # Data directories (as requested)
    - node_modules
    # ... (comprehensive list)

# Processing settings
processing:
  default_workers: 2              # Judge evaluation workers
  default_save_interval: 5        # Incremental saving interval
  max_raw_length: 2048           # Code segment length limit
  min_raw_length: 32             # Minimum segment length
```

## Usage

### Basic Usage

```bash
python -m x_spanformer.pipelines.repo2jsonl \
  -u https://github.com/user/repository \
  -i ./repos/ \
  -o ./output/
```

### With Specific Branch

```bash
python -m x_spanformer.pipelines.repo2jsonl \
  -u https://github.com/user/repository \
  -i ./repos/ \
  -o ./output/ \
  --branch develop
```

### Advanced Usage

```bash
python -m x_spanformer.pipelines.repo2jsonl \
  -u https://github.com/microsoft/vscode \
  -i ./repos/ \
  -o ./training_data/ \
  -n vscode_dataset \
  -w 4 \
  --save-interval 10 \
  --pretty \
  --force
```

## CLI Options

### Required Arguments

- `-u, --url`: GitHub repository URL (e.g., `https://github.com/user/repo`)
- `-i, --input`: Input directory to export repository files  
- `-o, --output`: Output directory for CSV and JSONL results

### Optional Arguments

- `-n, --name`: Base name for output files (default: `repo_dataset`)
- `-w, --workers`: Number of concurrent workers (default: from config)
- `--save-interval`: Save progress every N records (default: from config)
- `--force`: Force re-export and regeneration of all files
- `-p, --pretty`: Output pretty JSON file for inspection
- `--branch`: Specific branch to clone (default: repository default branch)

## How It Works

### Step 1: Repository Export
```bash
git clone --depth 1 --single-branch <url> <destination>
```
- Downloads only the latest commit (shallow clone)
- Targets specific branch if specified
- Removes `.git` directory after clone to avoid conflicts

### Step 2: File Discovery
- Recursively scans exported repository
- Filters by file extensions from config
- Skips directories listed in config (including `data/` as requested)

### Step 3: Content Extraction
- Reads text content from relevant files
- Creates CSV with content, path, language, and size
- Handles encoding issues gracefully

### Step 4: Judge Evaluation
- Uses shared `csv_processor.py` logic
- Same evaluation system as `pdf2jsonl`
- Outputs structured JSONL records

## Directory Structure

### Input Structure (After Export)
```
repos/
└── owner__repository/          # Clean source export (no .git)
    ├── src/
    ├── lib/
    ├── README.md
    └── ...
```

### Output Structure
```
output/
├── csv/
│   └── <repo_hash>.csv         # Extracted code content
├── jsonl/
│   ├── repo_dataset.jsonl      # Keep records
│   └── discard.jsonl           # Discard records
└── logs/
    └── repo2jsonl.log          # Processing logs
```

## File Filtering

### Included Extensions (From Config)
```
.py .js .ts .java .cpp .c .h .rs .go .rb .php .cs 
.swift .kt .scala .clj .hs .ml .r .sql .sh .bat 
.ps1 .yaml .yml .json .xml .md .rst .txt
```

### Excluded Directories (From Config)
- **Version Control**: `.git`, `.svn`, `.hg`
- **Build Artifacts**: `build/`, `dist/`, `target/`, `bin/`
- **Dependencies**: `node_modules/`, `vendor/`, `third_party/`
- **Data Directories**: `data/`, `datasets/`, `models/`, `checkpoints/` ⭐
- **Documentation**: `docs/`, `documentation/`
- **Cache/Temp**: `cache/`, `tmp/`, `logs/`
- **IDE/Editor**: `.vscode/`, `.idea/`, `.vs/`

## Shared Components

The pipeline reuses components from the existing codebase:

### `shared/csv_processor.py`
- Extracted from `pdf2jsonl.py`
- Handles judge evaluation pipeline
- Supports incremental saving and resume functionality

### `shared/repo_exporter.py`
- `GitRepoExporter`: Handles git clone operations
- `CodeFileExtractor`: Extracts and filters code files
- Both classes load configuration from `repo2jsonl.yaml`

### Integration with Existing Systems
- Uses `load_judge_config()` for LLM evaluation settings
- Uses `PretrainRecord` schema for output consistency
- Uses same logging and progress display system

## Examples

### Process Python Project
```bash
python -m x_spanformer.pipelines.repo2jsonl \
  -u https://github.com/psf/requests \
  -i ./repos/ \
  -o ./training_data/ \
  -n requests_dataset
```

### Process Specific Branch
```bash
python -m x_spanformer.pipelines.repo2jsonl \
  -u https://github.com/user/repo \
  -i ./repos/ \
  -o ./training_data/ \
  --branch feature/new-api \
  --force
```

### Large Multi-Language Project
```bash
python -m x_spanformer.pipelines.repo2jsonl \
  -u https://github.com/tensorflow/tensorflow \
  -i ./repos/ \
  -o ./training_data/ \
  -n tensorflow_dataset \
  -w 6
```

## Advantages of Git Export Approach

| Feature | Git Export | API Crawling | Full Clone |
|---------|------------|--------------|------------|
| Speed | ⚡ Fast | 🐌 Slow | ⚡ Fast |
| Rate Limits | ✅ None | ❌ Limited | ✅ None |
| API Keys | ✅ Not needed | ⚠️ Recommended | ✅ Not needed |
| Disk Usage | ✅ Source only | ✅ Minimal | ❌ Full history |
| Dependencies | ✅ Git only | ❌ Multiple | ✅ Git only |
| Reliability | ✅ High | ⚠️ Network dependent | ✅ High |
| Branch Support | ✅ Yes | ⚠️ Complex | ✅ Yes |
| Offline Processing | ✅ Yes | ❌ No | ✅ Yes |
| Configuration | ✅ YAML-driven | ⚠️ Code-based | ✅ YAML-driven |

## Error Handling

Robust error handling for:

- **Repository Access**: Clear messages for private/non-existent repos
- **Branch Issues**: Specific error for invalid branch names
- **Git Errors**: Detailed git command error reporting  
- **File Processing**: Continues with remaining files if individual files fail
- **Binary Detection**: Automatically skips binary and data files
- **Encoding Issues**: Graceful handling of text encoding problems
- **Judge Failures**: Retry logic for LLM evaluation errors

## Output Schema

Same `PretrainRecord` schema as other pipelines:

```json
{
  "raw": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "type": "code", 
  "id": {"id": "uuid4-generated"},
  "meta": {
    "status": "keep",
    "doc_language": "en",
    "extracted_by": "repo2jsonl", 
    "confidence": 0.91,
    "source_file": "https://github.com/user/repo#algorithms.py",
    "notes": "Clean Python implementation"
  }
}
```

## Dependencies

### Required Packages
```bash
pip install gitpython pandas rich pydantic asyncio pyyaml
```

### System Requirements
- Python 3.8+
- Git installed and accessible in PATH
- Internet connection for repository cloning
- Configured Judge LLM service (Ollama recommended)

## Integration with X-Spanformer

The pipeline follows the same patterns as existing pipelines:

- **Configuration**: Uses YAML config files like `jsonl2vocab.py`
- **Judge System**: Reuses same evaluation system as `pdf2jsonl.py`
- **Schema**: Outputs `PretrainRecord` format for consistency
- **Logging**: Uses same logging infrastructure as other pipelines
- **Shared Components**: Extracts reusable logic for other pipelines

## Notes

- The pipeline uses `git clone --depth 1` for efficiency (only latest commit)
- The `.git` directory is automatically removed to prevent conflicts
- All configuration is centralized in `repo2jsonl.yaml`
- Repository exports are cached and reused unless `--force` is specified
- Private repositories work if you have git access (SSH keys, etc.)
- Branch selection allows targeting specific development branches
- Clean source-only exports make processing much more efficient than full clones

This approach provides the perfect balance of simplicity, efficiency, reliability, and maintainability through configuration-driven design!
