"""
Tests for shared pipeline components
"""
import pytest
import tempfile
import csv
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from x_spanformer.pipelines.shared.repo_exporter import GitRepoExporter, CodeFileExtractor


class TestGitRepoExporter:
    """Test cases for GitRepoExporter."""
    
    def test_init_with_config(self):
        """Test initialization with provided config."""
        config = {
            "repository": {
                "clone_depth": 1,
                "single_branch": True,
                "remove_git_dir": True
            }
        }
        
        with patch('x_spanformer.pipelines.shared.repo_exporter.HAS_GITPYTHON', True):
            exporter = GitRepoExporter(config)
            assert exporter.config == config
    
    def test_init_without_gitpython(self):
        """Test initialization fails without GitPython."""
        with patch('x_spanformer.pipelines.shared.repo_exporter.HAS_GITPYTHON', False):
            with pytest.raises(ImportError, match="GitPython is required"):
                GitRepoExporter()
    
    def test_extract_repo_name(self):
        """Test repository name extraction from URLs."""
        config = {"repository": {}}
        
        with patch('x_spanformer.pipelines.shared.repo_exporter.HAS_GITPYTHON', True):
            exporter = GitRepoExporter(config)
            
            # Test different URL formats
            assert exporter._extract_repo_name("https://github.com/user/repo") == "user__repo"
            assert exporter._extract_repo_name("https://github.com/user/repo.git") == "user__repo"
            assert exporter._extract_repo_name("https://github.com/repo") == "repo"
    
    @patch('x_spanformer.pipelines.shared.repo_exporter.git')
    @patch('x_spanformer.pipelines.shared.repo_exporter.shutil')
    def test_export_repository_success(self, mock_shutil, mock_git):
        """Test successful repository export."""
        config = {
            "repository": {
                "clone_depth": 1,
                "single_branch": True,
                "remove_git_dir": True
            }
        }
        
        with patch('x_spanformer.pipelines.shared.repo_exporter.HAS_GITPYTHON', True):
            exporter = GitRepoExporter(config)
            
            # Mock git repo
            mock_repo = Mock()
            mock_repo.head.commit.hexsha = "abcd1234567890"
            mock_repo.active_branch.name = "main"
            mock_repo.head.commit.message = "Test commit message"
            mock_git.Repo.clone_from.return_value = mock_repo
            
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = Path(temp_dir)
                url = "https://github.com/test/repo"
                
                # Mock the git directory check
                git_dir_path = input_dir / "test__repo" / ".git"
                
                result = exporter.export_repository(url, input_dir, force=False)
                
                expected_path = input_dir / "test__repo"
                assert result == expected_path
                
                # Verify git clone was called
                mock_git.Repo.clone_from.assert_called_once_with(
                    url, str(expected_path), depth=1, single_branch=True
                )
    
    def test_is_valid_export(self):
        """Test validation of existing exports."""
        config = {"repository": {}}
        
        with patch('x_spanformer.pipelines.shared.repo_exporter.HAS_GITPYTHON', True):
            exporter = GitRepoExporter(config)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / "repo"
                
                # Test non-existent path
                assert not exporter._is_valid_export(repo_path)
                
                # Test directory with .git (invalid export)
                repo_path.mkdir()
                git_dir = repo_path / ".git"
                git_dir.mkdir()
                assert not exporter._is_valid_export(repo_path)
                
                # Test valid export (no .git, has files)
                git_dir.rmdir()
                (repo_path / "test.py").write_text("print('hello')")
                assert exporter._is_valid_export(repo_path)


class TestCodeFileExtractor:
    """Test cases for CodeFileExtractor."""
    
    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "repository": {
                "extensions": [".py", ".js", ".ts"],
                "max_file_size": 10000,
                "skip_directories": ["node_modules", "data"]
            }
        }
        
        extractor = CodeFileExtractor(config)
        assert extractor.extensions == {".py", ".js", ".ts"}
        assert extractor.max_file_size == 10000
        assert extractor.skip_directories == {"node_modules", "data"}
    
    def test_init_with_defaults(self):
        """Test initialization with default configuration."""
        extractor = CodeFileExtractor({})
        
        # Should have default values
        assert isinstance(extractor.extensions, set)
        assert extractor.max_file_size == 50000  # default
        assert isinstance(extractor.skip_directories, set)
        assert isinstance(extractor.binary_extensions, set)
    
    def test_find_code_files(self):
        """Test finding code files in repository."""
        config = {
            "repository": {
                "extensions": [".py", ".js"],
                "skip_directories": ["node_modules", "data"]
            }
        }
        
        extractor = CodeFileExtractor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create test files
            (repo_path / "main.py").write_text("print('hello')")
            (repo_path / "script.js").write_text("console.log('hello');")
            (repo_path / "readme.txt").write_text("This is readme")  # Not in extensions
            (repo_path / "binary.exe").write_text("binary data")  # Binary extension
            
            # Create files in skip directories
            skip_dir = repo_path / "data"
            skip_dir.mkdir()
            (skip_dir / "skip.py").write_text("should be skipped")
            
            files = extractor._find_code_files(repo_path)
            
            # Should only find .py and .js files, not in skip directories
            file_names = [f.name for f in files]
            assert "main.py" in file_names
            assert "script.js" in file_names
            assert "readme.txt" not in file_names  # Wrong extension
            assert "skip.py" not in file_names  # In skip directory
            assert "binary.exe" not in file_names  # Binary extension
    
    def test_extract_file_content(self):
        """Test extracting content from files."""
        config = {
            "repository": {
                "max_file_size": 100
            }
        }
        
        extractor = CodeFileExtractor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Test normal file
            normal_file = repo_path / "normal.py"
            normal_file.write_text("print('hello world')")
            content = extractor._extract_file_content(normal_file)
            assert content == "print('hello world')"
            
            # Test empty file
            empty_file = repo_path / "empty.py"
            empty_file.write_text("")
            content = extractor._extract_file_content(empty_file)
            assert content is None  # Empty files should be skipped
            
            # Test large file
            large_file = repo_path / "large.py"
            large_content = "x" * 1000  # Larger than max_file_size
            large_file.write_text(large_content)
            content = extractor._extract_file_content(large_file)
            assert content is None  # Should be skipped due to size
    
    def test_extract_to_csv(self):
        """Test extracting files to CSV format."""
        config = {
            "repository": {
                "extensions": [".py", ".js"],
                "max_file_size": 10000,
                "skip_directories": ["data"]
            }
        }
        
        extractor = CodeFileExtractor(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            repo_path.mkdir()
            output_dir = Path(temp_dir) / "output"
            
            # Create test files
            (repo_path / "main.py").write_text("def main():\n    print('hello')")
            (repo_path / "app.js").write_text("function app() {\n    console.log('hello');\n}")
            
            csv_files = extractor.extract_to_csv(repo_path, output_dir, force=True)
            
            assert len(csv_files) == 1
            csv_file = csv_files[0]
            assert csv_file.exists()
            
            # Check CSV content
            with csv_file.open('r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == ['content', 'file_path', 'language', 'size_chars']
                
                rows = list(reader)
                assert len(rows) == 2  # Two files
                
                # Check first row (order may vary due to sorting)
                contents = [row[0] for row in rows]
                assert "def main():" in contents[0] or "function app()" in contents[0]
    
    def test_detect_language(self):
        """Test language detection from file extensions."""
        extractor = CodeFileExtractor({})
        
        assert extractor._detect_language(".py") == "python"
        assert extractor._detect_language(".js") == "javascript"
        assert extractor._detect_language(".ts") == "typescript"
        assert extractor._detect_language(".java") == "java"
        assert extractor._detect_language(".unknown") == "unknown"
    
    def test_is_likely_binary(self):
        """Test binary content detection."""
        extractor = CodeFileExtractor({})
        
        # Text content should not be detected as binary
        assert not extractor._is_likely_binary("def hello():\n    print('world')")
        
        # Content with null bytes should be detected as binary
        assert extractor._is_likely_binary("hello\x00world")
        
        # Content with low printable ratio should be detected as binary
        non_printable = "".join(chr(i) for i in range(32)) * 10  # Control characters
        assert extractor._is_likely_binary(non_printable)


if __name__ == "__main__":
    pytest.main([__file__])
