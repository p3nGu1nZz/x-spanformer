"""
Test suite for x_spanformer.pipelines.jsonl2vocab module.

Tests vocabulary induction pipeline from JSONL files including:
- JSONL file discovery and loading
- Corpus construction from records  
- Vocabulary induction pipeline
- Output generation and statistics
- Error handling for invalid inputs
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.pipelines import jsonl2vocab
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.metadata import RecordMeta


class TestJsonl2VocabPipeline:
    """Test jsonl2vocab pipeline functionality."""
    
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.tmp_dir / "input" 
        self.output_dir = self.tmp_dir / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
    def teardown_method(self):
        # Close and remove all logging handlers to avoid file locks on Windows
        import logging
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
            
        # Also clean up logger-specific handlers
        for logger_name in ['jsonl2vocab']:
            logger_obj = logging.getLogger(logger_name)
            for handler in logger_obj.handlers[:]:
                handler.close()
                logger_obj.removeHandler(handler)
        
        import shutil
        import time
        if self.tmp_dir.exists():
            try:
                shutil.rmtree(self.tmp_dir)
            except PermissionError:
                # On Windows, sometimes we need to wait a bit for file handles to be released
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.tmp_dir)
                except PermissionError:
                    # If still fails, skip cleanup - test tmp dirs will be cleaned by OS
                    pass
    
    def test_parse_args_required(self):
        """Test argument parsing with required arguments."""
        with patch('sys.argv', ['jsonl2vocab', '-i', str(self.input_dir), '-o', str(self.output_dir)]):
            args = jsonl2vocab.parse_args()
            assert args.indir == self.input_dir
            assert args.outdir == self.output_dir
    
    def test_find_jsonl_files_single_file(self):
        """Test finding JSONL files in directory."""
        jsonl_file = self.input_dir / "test.jsonl"
        jsonl_file.touch()
        
        files = jsonl2vocab.find_jsonl_files(self.input_dir)
        
        assert len(files) == 1
        assert files[0] == jsonl_file
    
    def test_find_jsonl_files_multiple_files(self):
        """Test finding multiple JSONL files."""
        files = ["test1.jsonl", "test2.jsonl", "test3.jsonl"]
        for filename in files:
            (self.input_dir / filename).touch()
        
        found_files = jsonl2vocab.find_jsonl_files(self.input_dir)
        
        assert len(found_files) == 3
        filenames = [f.name for f in found_files]
        for filename in files:
            assert filename in filenames
    
    def test_find_jsonl_files_recursive(self):
        """Test recursive file discovery."""
        subdir = self.input_dir / "subdir"
        subdir.mkdir()
        
        (self.input_dir / "root.jsonl").touch()
        (subdir / "sub.jsonl").touch()
        
        files = jsonl2vocab.find_jsonl_files(self.input_dir)
        
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "root.jsonl" in filenames
        assert "sub.jsonl" in filenames
    
    def test_find_jsonl_files_empty_directory(self):
        """Test finding files in empty directory."""
        with pytest.raises(SystemExit):
            jsonl2vocab.find_jsonl_files(self.input_dir)
    
    def test_find_jsonl_files_nonexistent_directory(self):
        """Test finding files in nonexistent directory."""
        nonexistent = self.tmp_dir / "nonexistent"
        with pytest.raises(FileNotFoundError):
            jsonl2vocab.find_jsonl_files(nonexistent)
    
    def test_load_corpus_valid_records(self):
        """Test loading corpus from valid JSONL records."""
        jsonl_file = self.input_dir / "test.jsonl"
        
        # Create complete RecordMeta objects
        sample_records = [
            {
                "raw": "hello world",
                "meta": {
                    "status": "keep",
                    "source_file": "test.pdf",
                    "doc_language": "en",
                    "extracted_by": "test",
                    "confidence": 0.8,
                    "notes": "good quality",
                    "tags": []
                }
            },
            {
                "raw": "goodbye",
                "meta": {
                    "status": "discard", 
                    "source_file": "test.pdf",
                    "doc_language": "en",
                    "extracted_by": "test",
                    "confidence": 0.3,
                    "notes": "poor quality",
                    "tags": ["discard"]
                }
            }
        ]
        
        with open(jsonl_file, 'w') as f:
            for record in sample_records:
                f.write(json.dumps(record) + '\n')
        
        corpus = jsonl2vocab.load_corpus([jsonl_file])
        
        assert len(corpus) == 1  # Only the "keep" record should be included
        assert "hello world" in corpus
        # "goodbye" should be filtered out due to "discard" status
    
    def test_load_corpus_empty_text_filtered(self):
        """Test that empty/whitespace text is filtered out."""
        jsonl_file = self.input_dir / "test.jsonl"
        
        sample_records = [
            {
                "raw": "valid text",
                "meta": {
                    "status": "keep",
                    "source_file": "test.pdf",
                    "doc_language": "en",
                    "extracted_by": "test",
                    "confidence": 0.8,
                    "notes": "good",
                    "tags": []
                }
            },
            {
                "raw": "",  # Empty string
                "meta": {
                    "status": "keep",
                    "source_file": "test.pdf", 
                    "doc_language": "en",
                    "extracted_by": "test",
                    "confidence": 0.8,
                    "notes": "empty",
                    "tags": []
                }
            },
            {
                "raw": "   ",  # Whitespace only
                "meta": {
                    "status": "keep",
                    "source_file": "test.pdf",
                    "doc_language": "en", 
                    "extracted_by": "test",
                    "confidence": 0.8,
                    "notes": "whitespace",
                    "tags": []
                }
            }
        ]
        
        with open(jsonl_file, 'w') as f:
            for record in sample_records:
                f.write(json.dumps(record) + '\n')
        
        corpus = jsonl2vocab.load_corpus([jsonl_file])
        
        assert len(corpus) == 1
        assert "valid text" in corpus
    
    def test_load_corpus_invalid_json(self):
        """Test handling invalid JSON lines."""
        jsonl_file = self.input_dir / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"raw": "valid", "meta": {"status": "keep", "source_file": "test.pdf", "doc_language": "en", "extracted_by": "test", "confidence": 0.8, "notes": "good", "tags": []}}\n')
            f.write('invalid json line\n')  # This should be skipped
            f.write('{"raw": "also valid", "meta": {"status": "keep", "source_file": "test.pdf", "doc_language": "en", "extracted_by": "test", "confidence": 0.8, "notes": "good", "tags": []}}\n')
        
        corpus = jsonl2vocab.load_corpus([jsonl_file])
        
        # Should load valid records and skip invalid line
        assert len(corpus) == 2
        assert "valid" in corpus
        assert "also valid" in corpus
    
    def test_load_hparams_valid_file(self):
        """Test loading hyperparameters from valid YAML file."""
        config_file = self.tmp_dir / "config.yaml"
        config_data = {
            "L_max": 5,
            "M_candidates": 500,
            "min_piece_prob": 0.001
        }
        
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        hparams = jsonl2vocab.load_hparams(config_file)
        
        assert hparams["L_max"] == 5
        assert hparams["M_candidates"] == 500
        assert hparams["min_piece_prob"] == 0.001
    
    def test_load_hparams_nonexistent_file(self):
        """Test loading hyperparameters from nonexistent file."""
        nonexistent_file = self.tmp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            jsonl2vocab.load_hparams(nonexistent_file)
    
    @patch('x_spanformer.pipelines.jsonl2vocab.build_candidate_set')
    def test_build_candidate_set_with_output(self, mock_build_candidates):
        """Test candidate set building with output generation."""
        mock_build_candidates.return_value = (["a", "b", "hello"], Counter({"a": 5, "b": 3, "hello": 2}))
        
        corpus = ["hello", "ab"]
        U_0, freq = jsonl2vocab.build_candidate_set_with_output(corpus, 5, 10, self.output_dir)
        
        assert U_0 == ["a", "b", "hello"]
        assert freq == Counter({"a": 5, "b": 3, "hello": 2})
        
        # Verify the core function was called
        mock_build_candidates.assert_called_once_with(corpus, 5, 10)
        assert freq["a"] == 5
        assert freq["b"] == 3
        assert freq["hello"] == 2
        
        # Check output files were created
        assert (self.output_dir / "full_freq" / "full_freq.json").exists()
        assert (self.output_dir / "candidates" / "candidates.txt").exists()
        
        mock_build_candidates.assert_called_once_with(corpus, 5, 10)
    
    def test_save_vocab(self):
        """Test saving vocabulary to JSONL format."""
        vocab = ["a", "hello", "world"]
        probabilities = {"a": 0.3, "hello": 0.4, "world": 0.3}
        stats = {
            "total_pieces": 3,
            "baseline_ppl": 10.0,
            "final_ppl": 8.5,
            "oov_rate": 0.02,
            "em_iterations": 5,
            "pruned_pieces": 0
        }

        output_file = self.output_dir / "test_vocab.jsonl"
        jsonl2vocab.save_vocab(output_file, vocab, probabilities, stats)
        
        assert output_file.exists()
        
        # Read and verify content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3
            
            pieces = []
            for line in lines:
                record = json.loads(line)
                assert "piece" in record
                assert "prob" in record
                pieces.append(record["piece"])
            
            assert "a" in pieces
            assert "hello" in pieces
            assert "world" in pieces
    
    @patch('x_spanformer.pipelines.jsonl2vocab.induce_vocabulary')
    @patch('x_spanformer.pipelines.jsonl2vocab.validate_vocabulary_completeness')
    @patch('x_spanformer.pipelines.jsonl2vocab.build_candidate_set_with_output')
    @patch('x_spanformer.pipelines.jsonl2vocab.load_corpus')
    @patch('x_spanformer.pipelines.jsonl2vocab.find_jsonl_files')
    @patch('x_spanformer.pipelines.jsonl2vocab.load_hparams')
    def test_main_pipeline_success(self, mock_load_hparams, mock_find_files, mock_load_corpus,
                                 mock_build_candidates, mock_validate, mock_induce_vocab):
        """Test main pipeline execution success path."""
        # Setup mocks
        mock_load_hparams.return_value = {
            "L_max": 5, 
            "M_candidates": 10,
            "T_max_iters": 20,
            "min_piece_prob": 1e-6,
            "delta_perplexity": 0.01,
            "delta_oov": 0.001
        }
        mock_find_files.return_value = [self.input_dir / "test.jsonl"]
        mock_load_corpus.return_value = ["hello", "world"]
        mock_build_candidates.return_value = (
            ["h", "e", "l", "o", "w", "r", "d"], 
            Counter({"h": 5, "e": 4, "l": 3, "o": 2, "w": 2, "r": 2, "d": 1})  # Non-empty counter
        )
        mock_validate.return_value = None  # No exception means validation passed
        mock_induce_vocab.return_value = (
            ["h", "e", "l", "o", "w", "r", "d"],
            {"h": 0.1, "e": 0.1, "l": 0.2, "o": 0.1, "w": 0.1, "r": 0.1, "d": 0.3},
            {"total_pieces": 7, "baseline_ppl": 2.5, "final_ppl": 2.0}
        )
        
        with patch('sys.argv', ['jsonl2vocab', '-i', str(self.input_dir), '-o', str(self.output_dir)]):
            jsonl2vocab.main()
        
        # Verify all stages were called
        mock_find_files.assert_called_once_with(self.input_dir)
        mock_load_corpus.assert_called_once()
        mock_build_candidates.assert_called_once()
        mock_validate.assert_called_once()
        mock_induce_vocab.assert_called_once()
        
        # Verify output file was created
        assert (self.output_dir / "vocab.jsonl").exists()


class TestJsonl2VocabEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
    
    def test_load_corpus_permission_error(self):
        """Test handling of permission errors when loading corpus."""
        jsonl_file = self.tmp_dir / "test.jsonl"
        jsonl_file.touch()
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                jsonl2vocab.load_corpus([jsonl_file])
    
    def test_save_vocab_creates_directory(self):
        """Test that save_vocab creates output directory if it doesn't exist."""
        output_file = self.tmp_dir / "new_dir" / "vocab.jsonl"
        vocab = ["a", "b"]
        probabilities = {"a": 0.5, "b": 0.5}
        stats = {
            "total_pieces": 2,
            "baseline_ppl": 5.0,
            "final_ppl": 4.8,
            "oov_rate": 0.0,
            "em_iterations": 3,
            "pruned_pieces": 0
        }

        jsonl2vocab.save_vocab(output_file, vocab, probabilities, stats)
        
        assert output_file.exists()
        assert output_file.parent.exists()
    
    def test_load_corpus_multiple_files(self):
        """Test loading corpus from multiple files."""
        file1 = self.tmp_dir / "file1.jsonl"
        file2 = self.tmp_dir / "file2.jsonl"
        
        record_template = {
            "meta": {
                "status": "keep",
                "source_file": "test.pdf",
                "doc_language": "en",
                "extracted_by": "test", 
                "confidence": 0.8,
                "notes": "good",
                "tags": []
            }
        }
        
        # File 1
        with open(file1, 'w') as f:
            record1 = {"raw": "text from file 1", **record_template}
            f.write(json.dumps(record1) + '\n')
        
        # File 2  
        with open(file2, 'w') as f:
            record2 = {"raw": "text from file 2", **record_template}
            f.write(json.dumps(record2) + '\n')
        
        corpus = jsonl2vocab.load_corpus([file1, file2])
        
        assert len(corpus) == 2
        assert "text from file 1" in corpus
        assert "text from file 2" in corpus


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
