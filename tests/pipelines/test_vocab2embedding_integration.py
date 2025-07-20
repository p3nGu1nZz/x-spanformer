#!/usr/bin/env python3
"""
tests/pipelines/test_vocab2embedding_integration.py

Integration tests for the updated vocab2embedding pipeline with PretrainRecord support.
"""
import pytest
import json
import tempfile
import logging
from pathlib import Path
import numpy as np
import sys

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.pipelines.vocab2embedding import (
    load_corpus,
    Vocab2EmbeddingPipeline,
    main
)


class TestVocab2EmbeddingIntegration:
    """Integration tests for vocab2embedding pipeline."""
    
    @pytest.fixture
    def sample_pretrain_records(self):
        """Create sample PretrainRecord format data."""
        return [
            {
                "raw": "the quick brown fox jumps over the lazy dog",
                "type": "text",
                "id": "001",
                "meta": {"source": "sample.txt", "status": "validated"}
            },
            {
                "raw": "hello world this is a test sequence",
                "type": "text", 
                "id": "002",
                "meta": {"source": "sample.txt", "status": "validated"}
            },
            {
                "raw": "should be discarded",
                "type": "text",
                "id": "003", 
                "meta": {"source": "sample.txt", "status": "discard"}
            },
            {
                "raw": "",  # Empty raw field
                "type": "text",
                "id": "004",
                "meta": {"source": "sample.txt", "status": "validated"}
            },
            {
                "raw": "final valid sequence for testing",
                "type": "text",
                "id": "005", 
                "meta": {"source": "sample.txt", "status": "validated"}
            }
        ]
    
    @pytest.fixture
    def sample_vocab(self):
        """Create sample vocabulary data."""
        return [
            {"piece": "the", "probability": 0.05},
            {"piece": "quick", "probability": 0.01},
            {"piece": "brown", "probability": 0.01},
            {"piece": "fox", "probability": 0.01},
            {"piece": "jumps", "probability": 0.005},
            {"piece": "over", "probability": 0.01},
            {"piece": "lazy", "probability": 0.005},
            {"piece": "dog", "probability": 0.01},
            {"piece": "hello", "probability": 0.01},
            {"piece": "world", "probability": 0.01},
            {"piece": "this", "probability": 0.02},
            {"piece": "is", "probability": 0.03},
            {"piece": "a", "probability": 0.04},
            {"piece": "test", "probability": 0.01},
            {"piece": "sequence", "probability": 0.005},
            {"piece": "final", "probability": 0.005},
            {"piece": "valid", "probability": 0.005},
            {"piece": "for", "probability": 0.02},
            {"piece": "testing", "probability": 0.005},
            # Single character pieces
            {"piece": "t", "probability": 0.08},
            {"piece": "h", "probability": 0.06},
            {"piece": "e", "probability": 0.12},
            {"piece": "q", "probability": 0.001},
            {"piece": "u", "probability": 0.03},
            {"piece": " ", "probability": 0.15}  # Space
        ]
    
    @pytest.fixture 
    def sample_config(self):
        """Create sample configuration."""
        return {
            "embed_dim": 64,
            "tau_vocab": 1e-4,
            "tau_comp": 1e-6,
            "w_max": 32
        }
    
    def test_load_corpus_pretrain_records(self, sample_pretrain_records):
        """Test loading corpus from PretrainRecord format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = Path(temp_dir) / "dataset.jsonl"
            with open(input_file, 'w', encoding='utf-8') as f:
                for record in sample_pretrain_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Load corpus
            sequences = load_corpus(str(input_file))
            
            # Should load 3 valid sequences (skip discarded and empty)
            assert len(sequences) == 3
            assert "the quick brown fox jumps over the lazy dog" in sequences
            assert "hello world this is a test sequence" in sequences
            assert "final valid sequence for testing" in sequences
            assert "should be discarded" not in sequences  # Discarded
            assert "" not in sequences  # Empty
    
    def test_load_corpus_invalid_records(self):
        """Test handling of invalid records in corpus."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file with invalid records
            input_file = Path(temp_dir) / "invalid.jsonl"
            with open(input_file, 'w', encoding='utf-8') as f:
                # Valid record
                f.write(json.dumps({"raw": "valid text", "type": "text"}) + '\n')
                # Invalid JSON
                f.write('{"invalid": json\n')
                # Missing raw field
                f.write(json.dumps({"content": "wrong field", "type": "text"}) + '\n')
                # Non-dict record
                f.write(json.dumps("just a string") + '\n')
                # Another valid record
                f.write(json.dumps({"raw": "another valid text", "type": "text"}) + '\n')
            
            # Should handle gracefully
            sequences = load_corpus(str(input_file))
            assert len(sequences) == 2
            assert "valid text" in sequences
            assert "another valid text" in sequences
    
    def test_load_corpus_empty_file(self):
        """Test handling of empty corpus file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty file
            input_file = Path(temp_dir) / "empty.jsonl"
            input_file.touch()
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="No valid sequences found"):
                load_corpus(str(input_file))
    
    def test_load_corpus_nonexistent_file(self):
        """Test handling of non-existent corpus file."""
        with pytest.raises(FileNotFoundError):
            load_corpus("nonexistent_file.jsonl")
    
    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Initialize pipeline
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            
            assert pipeline.device == 'cpu'
            assert pipeline.config == sample_config
            assert pipeline.unigram_lm is None  # Not loaded yet
            assert pipeline.seed_embedder is None
            assert pipeline.conv_encoder is None
            assert pipeline.candidate_generator is None
    
    def test_pipeline_vocab_loading(self, sample_config, sample_vocab):
        """Test vocabulary loading in pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Create vocab file
            vocab_file = Path(temp_dir) / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in sample_vocab:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Initialize and load
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            pipeline.load_vocabulary(str(vocab_file))
            
            # Check components are initialized
            assert pipeline.unigram_lm is not None
            assert pipeline.seed_embedder is not None
            assert pipeline.conv_encoder is not None
            assert pipeline.candidate_generator is not None
            
            # Check vocabulary is loaded correctly
            assert pipeline.unigram_lm is not None
            assert len(pipeline.unigram_lm.piece_to_idx) == len(sample_vocab)
            assert "the" in pipeline.unigram_lm.piece_to_idx
            assert " " in pipeline.unigram_lm.piece_to_idx
    
    def test_pipeline_sequence_processing(self, sample_config, sample_vocab):
        """Test processing a single sequence through pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup pipeline
            config_file = Path(temp_dir) / "config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            vocab_file = Path(temp_dir) / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in sample_vocab:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            pipeline.load_vocabulary(str(vocab_file))
            
            # Process sequence
            test_sequence = "the quick brown"
            result = pipeline.process_sequence(test_sequence)
            
            # Check result structure
            assert 'soft_probabilities' in result
            assert 'seed_embeddings' in result
            assert 'contextual_embeddings' in result
            assert 'span_candidates' in result
            assert 'sequence_length' in result
            assert 'num_candidates' in result
            
            # Check shapes
            seq_len = len(test_sequence)
            vocab_size = len(sample_vocab)
            embed_dim = sample_config['embed_dim']
            
            assert result['soft_probabilities'].shape == (seq_len, vocab_size)
            assert result['seed_embeddings'].shape == (seq_len, embed_dim)
            assert result['contextual_embeddings'].shape == (seq_len, embed_dim)
            assert result['sequence_length'] == seq_len
            assert isinstance(result['span_candidates'], list)
            assert result['num_candidates'] == len(result['span_candidates'])
    
    def test_pipeline_error_handling(self, sample_config):
        """Test pipeline error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = Path(temp_dir) / "config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            
            # Try to process without loading vocabulary
            with pytest.raises(RuntimeError, match="Vocabulary not loaded"):
                pipeline.process_sequence("test sequence")
    
    def test_vocab_file_formats(self, sample_config):
        """Test different vocabulary file formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Test with 'prob' field instead of 'probability'
            vocab_with_prob = [
                {"piece": "the", "prob": 0.05},
                {"piece": "test", "prob": 0.01},
                {"piece": " ", "prob": 0.15}
            ]
            
            vocab_file = Path(temp_dir) / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in vocab_with_prob:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            pipeline = Vocab2EmbeddingPipeline(str(config_file), device='cpu')
            pipeline.load_vocabulary(str(vocab_file))
            
            # Should work with 'prob' field
            assert pipeline.unigram_lm is not None
            assert len(pipeline.unigram_lm.piece_to_idx) == 3
            assert "the" in pipeline.unigram_lm.piece_to_idx
    
    def cleanup_logging(self):
        """Clean up logging handlers to avoid Windows file locking."""
        import logging
        # Clear all handlers from the embedding logger
        logger_names = ['embedding_logger', 'x_spanformer.embedding.embedding_logging']
        for name in logger_names:
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        # Clear root logger handlers too
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if hasattr(handler, 'baseFilename'):
                handler.close()

    def test_main_function_integration(self, sample_pretrain_records, sample_vocab, sample_config):
        """Test the main function with PretrainRecord input."""
        # Use current directory to avoid Windows file locking issues with tempfile
        import shutil
        test_dir = Path('.') / 'temp_test_main_integration'
        
        try:
            # Clean up first if exists
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
            
            test_dir.mkdir(exist_ok=True)
            
            # Create input files
            dataset_file = test_dir / "dataset.jsonl"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                for record in sample_pretrain_records[:2]:  # Use first 2 records
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            vocab_file = test_dir / "vocab.jsonl"
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for entry in sample_vocab:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            config_file = test_dir / "config.yaml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            output_dir = test_dir / "output"
            
            # Mock command line arguments
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    'vocab2embedding.py',
                    '--vocab', str(vocab_file),
                    '--input', str(dataset_file),
                    '--output', str(output_dir),
                    '--config', str(config_file),
                    '--device', 'cpu',
                    '--max-length', '100'
                ]
                
                # Run main function
                main()
                
                # Check outputs exist
                assert output_dir.exists()
                assert (output_dir / "embedding.log").exists()
                
                # Should have processed 2 sequences
                embedding_files = list(output_dir.glob("embedding_*.json"))
                assert len(embedding_files) == 2
                
                # Check output files exist
                for i in [1, 2]:  # sequence IDs 1 and 2
                    seq_id_str = f"{i:06d}"
                    assert (output_dir / f"embedding_{seq_id_str}.json").exists()
                    assert (output_dir / f"soft_probs_{seq_id_str}.npy").exists()
                    assert (output_dir / f"seed_emb_{seq_id_str}.npy").exists()
                    assert (output_dir / f"context_emb_{seq_id_str}.npy").exists()
                
                # Check log file contains expected messages
                with open(output_dir / "embedding.log", 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    assert "X-SPANFORMER EMBEDDING GENERATION PIPELINE" in log_content
                    assert "Pipeline completed" in log_content
                    assert "Processing sequences from:" in log_content
                
            finally:
                sys.argv = original_argv
                self.cleanup_logging()
                
        finally:
            # Clean up test directory
            if test_dir.exists():
                self.cleanup_logging()
                import time
                time.sleep(0.1)  # Brief pause for Windows
                shutil.rmtree(test_dir, ignore_errors=True)
    
    def test_logging_integration(self, sample_pretrain_records):
        """Test that logging works correctly with PretrainRecord format."""
        import shutil
        # Use current directory to avoid Windows file locking issues  
        test_dir = Path('.') / 'temp_test_logs_integration'
        try:
            # Clean up first if exists
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
            
            test_dir.mkdir(exist_ok=True)
            
            # Create input file
            input_file = test_dir / "dataset.jsonl"
            with open(input_file, 'w', encoding='utf-8') as f:
                for record in sample_pretrain_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Set up logging directory (same as output directory)
            log_dir = test_dir
            
            # Import and setup logging
            from x_spanformer.embedding.embedding_logging import setup_embedding_logging
            logger = setup_embedding_logging(log_dir, 'test_integration')
            
            # Load corpus (should log statistics)
            sequences = load_corpus(str(input_file))
            
            # Check log file is in the root output directory
            log_file = log_dir / "embedding.log"
            assert log_file.exists()
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "STAGE 1: CORPUS LOADING" in content
                assert "Total input records: 5" in content
                assert "Valid sequences: 3" in content
                assert "Discarded sequences: 1" in content
                assert "Invalid records: 1" in content
                
        finally:
            self.cleanup_logging()
            # Clean up directory
            if test_dir.exists():
                import time
                time.sleep(0.1)  # Brief pause for Windows
                shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
