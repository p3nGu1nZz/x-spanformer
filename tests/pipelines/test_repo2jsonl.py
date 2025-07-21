"""
Tests for repo2jsonl.py pipeline
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from x_spanformer.pipelines import repo2jsonl
from x_spanformer.schema.pretrain_record import PretrainRecord


class TestRepo2JSONL:
    """Test cases for repo2jsonl pipeline."""
    
    def test_load_pipeline_config(self):
        """Test loading pipeline configuration."""
        # Test with existing config file
        config = repo2jsonl.load_pipeline_config()
        assert isinstance(config, dict)
        assert "repository" in config
        assert "processing" in config
        
        # Test configuration structure
        repo_config = config.get("repository", {})
        assert "extensions" in repo_config
        assert "skip_directories" in repo_config
        assert "max_file_size" in repo_config
        
        processing_config = config.get("processing", {})
        assert "default_workers" in processing_config
        assert "default_save_interval" in processing_config
    
    def test_parse_args(self):
        """Test argument parsing."""
        with patch('sys.argv', ['repo2jsonl.py', '-u', 'https://github.com/test/repo', '-i', './input', '-o', './output']):
            args = repo2jsonl.parse_args()
            assert args.url == 'https://github.com/test/repo'
            assert str(args.input) == 'input'  # Path normalizes ./input to input
            assert str(args.output) == 'output'  # Path normalizes ./output to output
            assert args.name == 'repo_dataset'  # default
            assert args.workers is None  # should use config default
            assert args.save_interval is None  # should use config default
    
    @patch("x_spanformer.pipelines.repo2jsonl.load_pipeline_config")
    @patch("x_spanformer.pipelines.repo2jsonl.setup_vocab_logging")
    @patch("x_spanformer.pipelines.repo2jsonl.check_ollama_connection")
    @patch("x_spanformer.pipelines.repo2jsonl.GitRepoExporter")
    @patch("x_spanformer.pipelines.repo2jsonl.CodeFileExtractor")
    @patch("x_spanformer.pipelines.repo2jsonl.process_all_csvs")
    @patch("x_spanformer.pipelines.repo2jsonl.load_existing_records")
    def test_run_pipeline(self, mock_load_existing, mock_process_csvs, 
                         mock_code_extractor_class, mock_repo_exporter_class,
                         mock_check_ollama, mock_setup_logging, mock_load_config):
        """Test complete pipeline run."""
        
        # Mock configuration
        mock_config = {
            "processing": {
                "default_workers": 2,
                "default_save_interval": 5
            },
            "repository": {
                "extensions": [".py", ".js"],
                "skip_directories": ["node_modules", "data"],
                "max_file_size": 50000
            }
        }
        mock_load_config.return_value = mock_config
        
        # Mock logging
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        
        # Mock Ollama connection
        async def mock_ollama_check(model):
            return True
        mock_check_ollama.return_value = mock_ollama_check("test")
        
        # Mock repo exporter
        mock_exporter = Mock()
        mock_repo_path = Path("/tmp/test_repo")
        mock_exporter.export_repository.return_value = mock_repo_path
        mock_repo_exporter_class.return_value = mock_exporter
        
        # Mock code extractor
        mock_extractor = Mock()
        mock_csv_files = [Path("/tmp/output/csv/test.csv")]
        mock_extractor.extract_to_csv.return_value = mock_csv_files
        mock_code_extractor_class.return_value = mock_extractor
        
        # Mock existing records and CSV processing
        mock_load_existing.return_value = []
        mock_records = [
            PretrainRecord(raw="def test(): pass", type="code"),
            PretrainRecord(raw="function test() {}", type="code")
        ]
        mock_process_csvs.return_value = mock_records
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            # Run the pipeline
            repo2jsonl.run(
                url="https://github.com/test/repo",
                input_dir=input_dir,
                output_dir=output_dir,
                name="test_dataset",
                workers=None,  # Should use config default
                save_interval=None,  # Should use config default
                force=False,
                pretty=False,
                branch=None
            )
            
            # Verify calls
            mock_load_config.assert_called_once()
            mock_setup_logging.assert_called_once_with(output_dir, 'repo2jsonl')
            
            # Verify repo exporter was called
            mock_repo_exporter_class.assert_called_once_with(mock_config)
            mock_exporter.export_repository.assert_called_once_with(
                "https://github.com/test/repo", input_dir, branch=None, force=False
            )
            
            # Verify code extractor was called
            mock_code_extractor_class.assert_called_once_with(mock_config)
            mock_extractor.extract_to_csv.assert_called_once_with(
                mock_repo_path, output_dir, force=False
            )
            
            # Verify CSV processing was called
            mock_process_csvs.assert_called_once()
            call_args = mock_process_csvs.call_args
            assert call_args[1]['w'] == 2  # default workers from config
            assert call_args[1]['save_interval'] == 5  # default save_interval from config
            assert call_args[1]['col'] == "content"
            assert call_args[1]['base_name'] == "test_dataset"
    
    @patch("x_spanformer.pipelines.repo2jsonl.load_pipeline_config")
    def test_run_with_config_override(self, mock_load_config):
        """Test that CLI arguments override config values."""
        mock_config = {
            "processing": {
                "default_workers": 2,
                "default_save_interval": 5
            }
        }
        mock_load_config.return_value = mock_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            with patch("x_spanformer.pipelines.repo2jsonl.setup_vocab_logging"), \
                 patch("x_spanformer.pipelines.repo2jsonl.check_ollama_connection") as mock_check_ollama, \
                 patch("x_spanformer.pipelines.repo2jsonl.GitRepoExporter"), \
                 patch("x_spanformer.pipelines.repo2jsonl.CodeFileExtractor"), \
                 patch("x_spanformer.pipelines.repo2jsonl.process_all_csvs") as mock_process_csvs, \
                 patch("x_spanformer.pipelines.repo2jsonl.load_existing_records"):
                
                # Mock Ollama connection
                async def mock_ollama_check_func(model):
                    return True
                mock_check_ollama.return_value = mock_ollama_check_func("test")
                
                mock_process_csvs.return_value = []
                
                # Run with CLI overrides
                repo2jsonl.run(
                    url="https://github.com/test/repo",
                    input_dir=input_dir,
                    output_dir=output_dir,
                    name="test_dataset",
                    workers=8,  # Override config
                    save_interval=10,  # Override config
                    force=False,
                    pretty=False,
                    branch=None
                )
                
                # Verify that CLI values were used
                call_args = mock_process_csvs.call_args
                assert call_args[1]['w'] == 8  # CLI override
                assert call_args[1]['save_interval'] == 10  # CLI override
    
    def test_load_existing_records_force_mode(self):
        """Test loading existing records with force mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create some dummy files
            jsonl_dir = output_dir / "jsonl"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_file = jsonl_dir / "test.jsonl"
            discard_file = jsonl_dir / "discard.jsonl"
            
            dataset_file.write_text('{"raw": "test", "type": "code"}\n')
            discard_file.write_text('{"raw": "bad", "type": "code"}\n')
            
            # Test force mode - should remove files and return empty list
            records = repo2jsonl.load_existing_records(output_dir, "test", force=True)
            assert records == []
            assert not dataset_file.exists()
            assert not discard_file.exists()
    
    def test_load_existing_records_normal_mode(self):
        """Test loading existing records in normal mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            jsonl_dir = output_dir / "jsonl"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a valid JSONL file with PretrainRecord format
            dataset_file = jsonl_dir / "test.jsonl"
            record_data = {
                "raw": "def test(): pass",
                "type": "code",
                "id": {"id": "test-123"},
                "meta": {
                    "status": "keep",
                    "doc_language": "en",
                    "extracted_by": "repo2jsonl",
                    "confidence": 0.8,
                    "source_file": "test.py"
                }
            }
            dataset_file.write_text(f'{record_data}\n'.replace("'", '"'))
            
            # Test normal mode - should load existing records
            records = repo2jsonl.load_existing_records(output_dir, "test", force=False)
            assert len(records) == 1
            assert records[0].raw == "def test(): pass"
            assert records[0].type == "code"
    
    def test_main_integration(self):
        """Test main function integration."""
        test_args = [
            'repo2jsonl.py',
            '-u', 'https://github.com/test/repo',
            '-i', './input',
            '-o', './output',
            '--name', 'test_dataset',
            '--workers', '4',
            '--save-interval', '10',
            '--force',
            '--pretty',
            '--branch', 'main'
        ]
        
        with patch('sys.argv', test_args), \
             patch('x_spanformer.pipelines.repo2jsonl.run') as mock_run:
            
            repo2jsonl.main()
            
            # Verify run was called with correct arguments
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]  # keyword arguments
            
            assert call_args['url'] == 'https://github.com/test/repo'
            assert str(call_args['input_dir']).endswith('input')
            assert str(call_args['output_dir']).endswith('output')
            assert call_args['name'] == 'test_dataset'
            assert call_args['workers'] == 4
            assert call_args['save_interval'] == 10
            assert call_args['force'] is True
            assert call_args['pretty'] is True
            assert call_args['branch'] == 'main'


if __name__ == "__main__":
    pytest.main([__file__])
