"""
Repository export and code file processing components using git clone --depth 1.
"""
import csv
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Set
from urllib.parse import urlparse
import re
import yaml

try:
    import git
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False

from x_spanformer.agents.rich_utils import console

class GitRepoExporter:
    """Export GitHub repositories using shallow git clone."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration from pipeline config."""
        if not HAS_GITPYTHON:
            raise ImportError(
                "GitPython is required for repository export. "
                "Install with: pip install gitpython"
            )
        
        # Load config from repo2jsonl.yaml
        if config is None:
            config_path = Path(__file__).parents[3] / "config" / "pipelines" / "repo2jsonl.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
        
        # Ensure config is not None
        self.config = config if config is not None else {}
        
    def export_repository(self, url: str, input_dir: Path, branch: Optional[str] = None, 
                         force: bool = False) -> Path:
        """
        Export a GitHub repository to the specified directory using shallow clone.
        Removes .git directory to avoid conflicts with parent repository.
        
        Args:
            url: GitHub repository URL
            input_dir: Base directory to export into
            branch: Specific branch to clone (None for default)
            force: Force re-export even if directory exists
            
        Returns:
            Path to the exported repository source files
        """
        # Parse repository name from URL
        repo_name = self._extract_repo_name(url)
        repo_path = input_dir / repo_name
        
        # Check if repository already exists
        if repo_path.exists() and not force:
            if self._is_valid_export(repo_path):
                console.print(f"[green]✔ Repository already exported: {repo_path}[/green]")
                console.print(f"[cyan]  Use --force to re-export[/cyan]")
                return repo_path
            else:
                console.print(f"[yellow]⚠ Export directory exists but appears incomplete. Re-exporting...[/yellow]")
                shutil.rmtree(repo_path)
        
        elif repo_path.exists() and force:
            console.print(f"[yellow]🔄 Force mode: Removing existing export[/yellow]")
            shutil.rmtree(repo_path)
        
        # Get configuration settings
        repo_config = self.config.get("repository", {})
        clone_depth = repo_config.get("clone_depth", 1)
        single_branch = repo_config.get("single_branch", True)
        remove_git_dir = repo_config.get("remove_git_dir", True)
        
        # Clone the repository with depth 1 (shallow clone)
        console.print(f"[cyan]📦 Exporting repository: {url}[/cyan]")
        if branch:
            console.print(f"[dim]  Branch: {branch}[/dim]")
        console.print(f"[dim]  Destination: {repo_path}[/dim]")
        console.print(f"[dim]  Depth: {clone_depth} | Single branch: {single_branch}[/dim]")
        
        try:
            # Prepare clone options
            clone_kwargs = {
                'depth': clone_depth,
                'single_branch': single_branch,
            }
            
            if branch:
                clone_kwargs['branch'] = branch
            
            # Clone the repository
            repo = git.Repo.clone_from(url, str(repo_path), **clone_kwargs)
            
            # Get some info before removing .git
            try:
                commit_sha = repo.head.commit.hexsha[:8]
                branch_name = repo.active_branch.name
                commit_message = str(repo.head.commit.message).strip().split('\n')[0][:60]
                console.print(f"[green]✅ Successfully cloned repository[/green]")
                console.print(f"[dim]  Latest commit: {commit_sha} on {branch_name}[/dim]")
                console.print(f"[dim]  Message: {commit_message}...[/dim]")
            except Exception:
                console.print(f"[green]✅ Successfully cloned repository[/green]")
            
            # Remove .git directory to avoid conflicts with parent repository
            if remove_git_dir:
                git_dir = repo_path / ".git"
                if git_dir.exists():
                    console.print(f"[cyan]🗑️  Removing .git directory to avoid conflicts[/cyan]")
                    shutil.rmtree(git_dir)
                    console.print(f"[green]✔ Clean source export complete[/green]")
            
            # Validate the export
            file_count = sum(1 for _ in repo_path.rglob('*') if _.is_file())
            console.print(f"[dim]  Exported {file_count:,} files[/dim]")
            
            return repo_path
            
        except Exception as e:
            error_msg = str(e)
            if "Repository not found" in error_msg or "not found" in error_msg.lower():
                raise RuntimeError(f"Repository not found or not accessible: {url}")
            elif "branch" in error_msg.lower() and branch:
                raise RuntimeError(f"Branch '{branch}' not found in repository: {url}")
            else:
                raise RuntimeError(f"Git clone failed: {e}")
    
    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from GitHub URL."""
        # Handle different GitHub URL formats
        if url.endswith('.git'):
            url = url[:-4]
        
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            # Use format: owner__repository for uniqueness
            return f"{path_parts[-2]}__{path_parts[-1]}"
        elif len(path_parts) == 1:
            return path_parts[0]
        else:
            # Fallback to hash of URL
            return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def _is_valid_export(self, repo_path: Path) -> bool:
        """Check if an existing directory appears to be a valid repository export."""
        if not repo_path.exists() or not repo_path.is_dir():
            return False
        
        # Check if it contains .git (should not for a clean export)
        if (repo_path / ".git").exists():
            return False
        
        # Check if it has any files (not just directories)
        has_files = any(f.is_file() for f in repo_path.rglob('*'))
        return has_files

class CodeFileExtractor:
    """Extract code files from exported repository and convert to CSV format."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration from pipeline config."""
        # Load config from repo2jsonl.yaml
        if config is None:
            config_path = Path(__file__).parents[3] / "config" / "pipelines" / "repo2jsonl.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
        
        # Ensure config is not None
        self.config = config if config is not None else {}
        repo_config = self.config.get("repository", {})
        
        # Get configuration values
        self.extensions = set(ext.lower() for ext in repo_config.get("extensions", [".py", ".js", ".ts"]))
        self.max_file_size = repo_config.get("max_file_size", 50000)
        self.skip_directories = set(repo_config.get("skip_directories", []))
        
        # Binary file extensions to skip
        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite', '.sqlite3',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.tiff', '.svg', '.webp',
            '.mp3', '.mp4', '.wav', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt',
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',
            '.woff', '.woff2', '.ttf', '.eot', '.otf',
            '.class', '.jar', '.pyc', '.pyo', '.o', '.obj', '.a', '.lib',
            '.pickle', '.pkl', '.npy', '.npz', '.h5', '.hdf5', '.parquet'
        }
        
    def extract_to_csv(self, repo_path: Path, output_dir: Path, force: bool = False) -> List[Path]:
        """
        Extract code files from exported repository and save as CSV files.
        
        Args:
            repo_path: Path to exported repository
            output_dir: Output directory for CSV files
            force: Force regeneration of CSV files
            
        Returns:
            List of generated CSV file paths
        """
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate hash-based CSV filename
        repo_hash = hashlib.sha256(str(repo_path).encode()).hexdigest()[:8]
        csv_file = csv_dir / f"{repo_hash}.csv"
        
        # Check if CSV already exists
        if csv_file.exists() and not force:
            console.print(f"[green]✔ Using existing CSV: {csv_file.name}[/green]")
            return [csv_file]
        
        console.print(f"[cyan]📄 Extracting code files from: {repo_path.name}[/cyan]")
        
        # Find all relevant files
        files_found = self._find_code_files(repo_path)
        
        if not files_found:
            console.print(f"[yellow]⚠ No code files found with specified extensions[/yellow]")
            console.print(f"[dim]  Extensions: {', '.join(sorted(self.extensions))}[/dim]")
            return []
        
        console.print(f"[green]✔ Found {len(files_found)} code files[/green]")
        
        # Extract file contents and create CSV
        extracted_count = 0
        skipped_count = 0
        
        with csv_file.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['content', 'file_path', 'language', 'size_chars'])
            
            for file_path in files_found:
                try:
                    content = self._extract_file_content(file_path)
                    if content is None:
                        skipped_count += 1
                        continue
                    
                    # Get relative path from repo root
                    rel_path = file_path.relative_to(repo_path)
                    
                    # Determine language from extension
                    language = self._detect_language(file_path.suffix)
                    
                    writer.writerow([
                        content,
                        str(rel_path),
                        language,
                        len(content)
                    ])
                    
                    extracted_count += 1
                    
                    if extracted_count % 50 == 0:
                        console.print(f"[dim]  Processed {extracted_count} files...[/dim]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠ Error processing {file_path}: {e}[/yellow]")
                    skipped_count += 1
                    continue
        
        console.print(f"[green]✅ Extracted {extracted_count} files to CSV[/green]")
        if skipped_count > 0:
            console.print(f"[yellow]  Skipped {skipped_count} files (too large, binary, or errors)[/yellow]")
        
        return [csv_file] if extracted_count > 0 else []
    
    def _find_code_files(self, repo_path: Path) -> List[Path]:
        """Find all code files matching the specified extensions."""
        files_found = []
        
        for file_path in repo_path.rglob('*'):
            # Skip if it's a directory
            if file_path.is_dir():
                continue
            
            # Skip if parent directory should be ignored
            if any(skip_dir in file_path.parts for skip_dir in self.skip_directories):
                continue
            
            # Skip binary files
            if file_path.suffix.lower() in self.binary_extensions:
                continue
            
            # Check if extension matches
            if file_path.suffix.lower() in self.extensions:
                files_found.append(file_path)
        
        return sorted(files_found)
    
    def _extract_file_content(self, file_path: Path) -> Optional[str]:
        """
        Extract content from a code file.
        
        Returns:
            File content as string, or None if file should be skipped
        """
        try:
            # Check file size first
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size * 2:  # Rough estimate (bytes vs chars)
                return None
            
            # Try to read as text file
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip if content is too large
            if len(content) > self.max_file_size:
                return None
            
            # Skip if content appears to be binary
            if self._is_likely_binary(content):
                return None
            
            # Skip empty files
            if not content.strip():
                return None
                
            return content
            
        except Exception:
            return None
    
    def _is_likely_binary(self, content: str) -> bool:
        """Check if content is likely binary data."""
        # Check for null bytes
        if '\x00' in content:
            return True
        
        # Check for high ratio of non-printable characters
        if len(content) > 100:
            sample = content[:1000]  # Check first 1000 chars
            printable_chars = sum(1 for c in sample if c.isprintable() or c.isspace())
            if len(sample) > 0:
                printable_ratio = printable_chars / len(sample)
                return printable_ratio < 0.7
        
        return False
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c', '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.clj': 'clojure',
            '.hs': 'haskell',
            '.ml': 'ocaml',
            '.r': 'r',
            '.sql': 'sql',
            '.sh': 'bash', '.bash': 'bash',
            '.bat': 'batch', '.cmd': 'batch',
            '.ps1': 'powershell',
            '.yaml': 'yaml', '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown',
            '.rst': 'rst',
            '.txt': 'text'
        }
        
        return language_map.get(extension.lower(), 'unknown')
