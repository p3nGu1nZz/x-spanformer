"""
Test suite for x_spanformer.schema.vocab module.

Tests vocabulary schemas including:
- VocabPiece model validation
- VocabStats model validation
- Field constraints and validation
- JSON serialization/deserialization
- Edge cases and error conditions
"""

import pytest
import json
import sys
from pathlib import Path
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.schema.vocab import VocabPiece, VocabStats


class TestVocabPiece:
    """Test VocabPiece schema."""
    
    def test_vocab_piece_valid_creation(self):
        """Test creating valid VocabPiece instances."""
        piece = VocabPiece(piece="hello", prob=0.5)
        
        assert piece.piece == "hello"
        assert piece.prob == 0.5
    
    def test_vocab_piece_min_max_probability(self):
        """Test probability bounds validation."""
        # Valid probabilities
        piece1 = VocabPiece(piece="a", prob=0.000001)  # Very small but > 0
        piece2 = VocabPiece(piece="b", prob=1.0)       # Maximum probability
        
        assert piece1.prob == 0.000001
        assert piece2.prob == 1.0
        
        # Invalid: zero probability
        with pytest.raises(ValidationError, match="greater than 0"):
            VocabPiece(piece="zero", prob=0.0)
        
        # Invalid: negative probability  
        with pytest.raises(ValidationError, match="greater than 0"):
            VocabPiece(piece="negative", prob=-0.1)
        
        # Invalid: probability > 1
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            VocabPiece(piece="large", prob=1.5)
    
    def test_vocab_piece_empty_piece(self):
        """Test VocabPiece with empty string piece."""
        # Empty string should be valid (represents empty substrings)
        piece = VocabPiece(piece="", prob=0.1)
        assert piece.piece == ""
        assert piece.prob == 0.1
    
    def test_vocab_piece_special_characters(self):
        """Test VocabPiece with special characters."""
        special_pieces = [
            "\n",      # newline
            "\t",      # tab
            " ",       # space
            "🎉",      # emoji
            "café",    # accented characters
            "αβγ",     # unicode
            "中文",     # Chinese characters
        ]
        
        for i, piece_text in enumerate(special_pieces):
            piece = VocabPiece(piece=piece_text, prob=0.1 + i * 0.01)
            assert piece.piece == piece_text
            assert piece.prob == 0.1 + i * 0.01
    
    def test_vocab_piece_json_serialization(self):
        """Test JSON serialization/deserialization."""
        piece = VocabPiece(piece="test", prob=0.42)
        
        # Serialize to JSON
        json_str = piece.model_dump_json()
        json_data = json.loads(json_str)
        
        assert json_data["piece"] == "test"
        assert json_data["prob"] == 0.42
        
        # Deserialize from JSON
        piece_restored = VocabPiece.model_validate(json_data)
        assert piece_restored.piece == piece.piece
        assert piece_restored.prob == piece.prob
    
    def test_vocab_piece_validation_error_messages(self):
        """Test that validation errors have clear messages."""
        # Missing piece
        with pytest.raises(ValidationError) as exc_info:
            VocabPiece(prob=0.5)  # type: ignore
        assert "piece" in str(exc_info.value)
        
        # Missing prob
        with pytest.raises(ValidationError) as exc_info:
            VocabPiece(piece="test")  # type: ignore
        assert "prob" in str(exc_info.value)
        
        # Invalid prob type
        with pytest.raises(ValidationError) as exc_info:
            VocabPiece(piece="test", prob="not_a_number")  # type: ignore
        assert "prob" in str(exc_info.value)


class TestVocabStats:
    """Test VocabStats schema."""
    
    def test_vocab_stats_valid_creation(self):
        """Test creating valid VocabStats instances."""
        stats = VocabStats(
            total_pieces=1000,
            baseline_ppl=50.0,
            final_ppl=45.0,
            oov_rate=0.01,
            em_iterations=5
        )
        
        assert stats.total_pieces == 1000
        assert stats.baseline_ppl == 50.0
        assert stats.final_ppl == 45.0
        assert stats.oov_rate == 0.01
        assert stats.em_iterations == 5
        assert stats.pruned_pieces == 0  # Default value
    
    def test_vocab_stats_with_pruned_pieces(self):
        """Test VocabStats with explicit pruned_pieces."""
        stats = VocabStats(
            total_pieces=800,
            baseline_ppl=60.0,
            final_ppl=55.0,
            oov_rate=0.005,
            em_iterations=8,
            pruned_pieces=200
        )
        
        assert stats.pruned_pieces == 200
    
    def test_vocab_stats_oov_rate_bounds(self):
        """Test OOV rate validation bounds."""
        # Valid OOV rates
        stats1 = VocabStats(
            total_pieces=1000, baseline_ppl=50.0, final_ppl=45.0,
            oov_rate=0.0, em_iterations=5  # Minimum
        )
        assert stats1.oov_rate == 0.0
        
        stats2 = VocabStats(
            total_pieces=1000, baseline_ppl=50.0, final_ppl=45.0,
            oov_rate=1.0, em_iterations=5  # Maximum
        )
        assert stats2.oov_rate == 1.0
        
        # Invalid: negative OOV rate
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            VocabStats(
                total_pieces=1000, baseline_ppl=50.0, final_ppl=45.0,
                oov_rate=-0.1, em_iterations=5
            )
        
        # Invalid: OOV rate > 1
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            VocabStats(
                total_pieces=1000, baseline_ppl=50.0, final_ppl=45.0,
                oov_rate=1.5, em_iterations=5
            )
    
    def test_vocab_stats_negative_values(self):
        """Test handling of negative values in fields that should be positive."""
        # Negative total_pieces should be invalid
        with pytest.raises(ValidationError):
            VocabStats(
                total_pieces=-1, baseline_ppl=50.0, final_ppl=45.0,
                oov_rate=0.01, em_iterations=5
            )
        
        # Negative em_iterations should be invalid
        with pytest.raises(ValidationError):
            VocabStats(
                total_pieces=1000, baseline_ppl=50.0, final_ppl=45.0,
                oov_rate=0.01, em_iterations=-1
            )
    
    def test_vocab_stats_json_serialization(self):
        """Test JSON serialization/deserialization."""
        stats = VocabStats(
            total_pieces=8192,
            baseline_ppl=245.7,
            final_ppl=251.3,
            oov_rate=0.001,
            em_iterations=5,
            pruned_pieces=1234
        )
        
        # Serialize to JSON
        json_str = stats.model_dump_json()
        json_data = json.loads(json_str)
        
        assert json_data["total_pieces"] == 8192
        assert json_data["baseline_ppl"] == 245.7
        assert json_data["final_ppl"] == 251.3
        assert json_data["oov_rate"] == 0.001
        assert json_data["em_iterations"] == 5
        assert json_data["pruned_pieces"] == 1234
        
        # Deserialize from JSON
        stats_restored = VocabStats.model_validate(json_data)
        assert stats_restored.total_pieces == stats.total_pieces
        assert stats_restored.baseline_ppl == stats.baseline_ppl
        assert stats_restored.final_ppl == stats.final_ppl
        assert stats_restored.oov_rate == stats.oov_rate
        assert stats_restored.em_iterations == stats.em_iterations
        assert stats_restored.pruned_pieces == stats.pruned_pieces
    
    def test_vocab_stats_realistic_values(self):
        """Test VocabStats with realistic values from vocabulary induction."""
        # Small vocabulary scenario
        small_stats = VocabStats(
            total_pieces=256,
            baseline_ppl=15.2,
            final_ppl=14.8,
            oov_rate=0.0,
            em_iterations=3
        )
        assert small_stats.total_pieces == 256
        
        # Large vocabulary scenario
        large_stats = VocabStats(
            total_pieces=32768,
            baseline_ppl=500.0,
            final_ppl=485.5,
            oov_rate=0.0001,
            em_iterations=10,
            pruned_pieces=5000
        )
        assert large_stats.total_pieces == 32768
        assert large_stats.pruned_pieces == 5000


class TestVocabSchemaEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_vocab_piece_float_precision(self):
        """Test VocabPiece with high precision float values."""
        # Very precise probability
        piece = VocabPiece(piece="precise", prob=0.123456789012345)
        
        # Should maintain reasonable precision
        assert abs(piece.prob - 0.123456789012345) < 1e-10
    
    def test_vocab_stats_extreme_perplexity_values(self):
        """Test VocabStats with extreme perplexity values."""
        # Very low perplexity (very good model)
        low_ppl = VocabStats(
            total_pieces=1000, baseline_ppl=1.1, final_ppl=1.05,
            oov_rate=0.0, em_iterations=20
        )
        assert low_ppl.baseline_ppl == 1.1
        assert low_ppl.final_ppl == 1.05
        
        # Very high perplexity (poor model)
        high_ppl = VocabStats(
            total_pieces=100, baseline_ppl=10000.0, final_ppl=9999.0,
            oov_rate=0.5, em_iterations=1
        )
        assert high_ppl.baseline_ppl == 10000.0
        assert high_ppl.final_ppl == 9999.0
    
    def test_vocab_piece_unicode_edge_cases(self):
        """Test VocabPiece with various Unicode edge cases."""
        unicode_cases = [
            "\u0000",      # null character
            "\u200B",      # zero-width space
            "\U0001F600",  # emoji
            "a\u0300",     # combining character
            "\uFEFF",      # BOM
        ]
        
        for i, piece_text in enumerate(unicode_cases):
            piece = VocabPiece(piece=piece_text, prob=0.1 + i * 0.01)
            assert piece.piece == piece_text
            
            # Should be serializable to JSON
            json_str = piece.model_dump_json()
            restored = VocabPiece.model_validate_json(json_str)
            assert restored.piece == piece_text
    
    def test_vocab_stats_zero_values(self):
        """Test VocabStats with zero values where appropriate."""
        # Zero pruned pieces should be valid (default)
        stats = VocabStats(
            total_pieces=1000, baseline_ppl=50.0, final_ppl=45.0,
            oov_rate=0.0, em_iterations=1, pruned_pieces=0
        )
        assert stats.pruned_pieces == 0
        
        # Zero OOV rate should be valid (perfect coverage)
        assert stats.oov_rate == 0.0
    
    def test_both_schemas_with_missing_fields(self):
        """Test both schemas handle missing required fields properly."""
        # VocabPiece missing piece
        with pytest.raises(ValidationError) as exc_info:
            VocabPiece(prob=0.5)  # type: ignore
        error_details = str(exc_info.value)
        assert "piece" in error_details and "Field required" in error_details
        
        # VocabStats missing required fields
        with pytest.raises(ValidationError) as exc_info:
            VocabStats(total_pieces=1000)  # type: ignore
        error_details = str(exc_info.value)
        assert "baseline_ppl" in error_details or "final_ppl" in error_details


class TestVocabSchemaIntegration:
    """Integration tests for vocabulary schemas."""
    
    def test_realistic_vocabulary_record(self):
        """Test creating a realistic vocabulary with multiple pieces and stats."""
        # Create vocabulary pieces
        pieces = [
            VocabPiece(piece="the", prob=0.05),
            VocabPiece(piece="of", prob=0.03),
            VocabPiece(piece="to", prob=0.025),
            VocabPiece(piece="a", prob=0.02),
            VocabPiece(piece="in", prob=0.015),
        ]
        
        # All pieces should be valid
        total_prob = sum(p.prob for p in pieces)
        assert 0 < total_prob < 1  # Partial vocabulary
        
        # Create corresponding stats
        stats = VocabStats(
            total_pieces=len(pieces),
            baseline_ppl=100.0,
            final_ppl=85.0,
            oov_rate=0.02,
            em_iterations=7,
            pruned_pieces=50
        )
        
        # Stats should reflect the vocabulary
        assert stats.total_pieces == len(pieces)
        assert stats.final_ppl < stats.baseline_ppl  # Improvement
    
    def test_complete_vocab_serialization(self):
        """Test serializing complete vocabulary data structure."""
        vocab_data = {
            "pieces": [
                VocabPiece(piece="hello", prob=0.01),
                VocabPiece(piece="world", prob=0.008),
                VocabPiece(piece=" ", prob=0.15),  # Space is common
            ],
            "stats": VocabStats(
                total_pieces=3,
                baseline_ppl=20.0,
                final_ppl=18.5,
                oov_rate=0.001,
                em_iterations=4
            )
        }
        
        # Convert to JSON-serializable format
        serializable = {
            "pieces": [p.model_dump() for p in vocab_data["pieces"]],
            "stats": vocab_data["stats"].model_dump()
        }
        
        # Should be serializable
        json_str = json.dumps(serializable)
        restored_data = json.loads(json_str)
        
        # Should be deserializable
        restored_pieces = [VocabPiece.model_validate(p) for p in restored_data["pieces"]]
        restored_stats = VocabStats.model_validate(restored_data["stats"])
        
        # Should match original
        assert len(restored_pieces) == 3
        assert restored_stats.total_pieces == 3
        assert restored_stats.final_ppl == 18.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
