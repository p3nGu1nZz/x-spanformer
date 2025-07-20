import unittest
import sys
from pathlib import Path
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x_spanformer.schema.xbar import XPSpan, CodeLabel, CodeRole, NaturalLabel, NaturalRole, HybridRole, XBarRole
from x_spanformer.schema.metadata import RecordMeta
from x_spanformer.schema.pretrain_record import PretrainRecord
from x_spanformer.schema.tokenization import TokenizedInput
from x_spanformer.schema.typing import RecordType
from x_spanformer.schema.span import SpanLabel
from x_spanformer.schema import (
    identifier, metadata, pretrain_record, scoring, source, span, status,
    tokenization, typing, validation, xbar
)


class TestSchemaComprehensive(unittest.TestCase):
    
    # Merged from test_schema.py
    def test_record_id(self):
        rec_id = identifier.RecordID()
        self.assertIsInstance(rec_id.id, str)
        with self.assertRaises(ValidationError):
            rec_id.id = "new-id"

    def test_record_meta(self):
        meta = metadata.RecordMeta(
            tags=["test"],
            doc_language="en",
            confidence=0.5,
            extracted_by="test",
            source_file="test.txt",
            notes="test note",
        )
        self.assertEqual(meta.tags, ["test"])
        self.assertEqual(meta.doc_language, "en")
        self.assertEqual(meta.confidence, 0.5)
        
        # Test with minimal data (all fields are optional except tags which has default)
        meta_minimal = metadata.RecordMeta(
            doc_language=None,
            extracted_by=None, 
            confidence=None,
            source_file=None,
            notes=None
        )
        self.assertEqual(meta_minimal.tags, [])
        self.assertIsNone(meta_minimal.confidence)

    def test_pretrain_record(self):
        record = pretrain_record.PretrainRecord(raw="test text")
        self.assertEqual(record.raw, "test text")
        self.assertIsInstance(record.id, identifier.RecordID)
        self.assertIsInstance(record.meta, metadata.RecordMeta)

    def test_span_label(self):
        label = span.SpanLabel(span=(0, 1), label="noun", text="test", role="token")
        self.assertEqual(label.span, (0, 1))
        self.assertEqual(label.label, "noun")

    def test_xp_span(self):
        xp = xbar.XPSpan(
            span=(0, 2), category="XP", role="specifier", label="NP", text="test"
        )
        self.assertEqual(xp.category, "XP")
        with self.assertRaises(ValidationError):
            xbar.XPSpan(
                span=(0, 1),
                category="invalid",  # type: ignore
                role="none",  # type: ignore
                label="none",
                text="test",
            )

    def test_tokenized_input(self):
        tok_input = tokenization.TokenizedInput(input=["a", "b"], tokenizer="test")
        self.assertEqual(tok_input.input, ["a", "b"])
        self.assertTrue(tok_input.preserve_whitespace)

    def test_record_type(self):
        rec_type = typing.RecordType(type="code")
        self.assertEqual(rec_type.type, "code")
        with self.assertRaises(ValidationError):
            typing.RecordType(type="invalid")  # type: ignore

    def test_source_info(self):
        src = source.SourceInfo(
            filename="test.txt",
            page_number=1,
            document_id="doc1",
            filetype="txt",
            line_number=1,
            section="body",
            source_url="http://example.com",
        )
        self.assertEqual(src.filename, "test.txt")
        self.assertEqual(src.page_number, 1)

    def test_record_status(self):
        stat = status.RecordStatus(stages=["csv_ingested", "validated"])
        self.assertEqual(stat.stages, ["csv_ingested", "validated"])
        with self.assertRaises(ValidationError):
            status.RecordStatus(stages=["invalid_stage"])  # type: ignore

    def test_entropy_profile(self):
        profile = scoring.EntropyProfile(
            token_entropy=1.5,
            span_overlap=0.5,
            structure_variance=0.5,
            fluency_score=0.5,
        )
        self.assertEqual(profile.token_entropy, 1.5)

    def test_validation_result(self):
        issue = validation.ValidationIssue(field="test", message="error", severity="error")
        res = validation.ValidationResult(is_valid=False, issues=[issue])
        self.assertFalse(res.is_valid)
        self.assertEqual(len(res.issues), 1)
    
    def test_xbar_with_all_roles(self):
        """Test XPSpan with various role types"""
        # Test with XBar role
        span1 = XPSpan(
            span=(0, 2),
            category="XP",
            role="specifier",
            label="NP",
            text="the fox"
        )
        self.assertEqual(span1.role, "specifier")
        
        # Test with all XBar roles - need to provide required fields
        xbar_roles: list[XBarRole] = ["specifier", "complement", "adjunct", "head", "modifier", "determiner", "nucleus"]
        for role in xbar_roles:
            span = XPSpan(span=(0, 1), category="X⁰", role=role, label="test", text="test")
            self.assertEqual(span.role, role)
    
    def test_record_meta_with_tags(self):
        """Test RecordMeta with various tag configurations"""
        # Empty tags (all fields are optional)
        meta1 = RecordMeta()  # type: ignore
        self.assertEqual(meta1.tags, [])
        
        # Multiple tags
        meta2 = RecordMeta(
            tags=["keep", "high_quality", "code"],
            doc_language="en",
            extracted_by="pdf2seg",
            confidence=0.95,
            source_file="test.pdf",
            notes="Excellent code example"
        )
        self.assertEqual(len(meta2.tags), 3)
        self.assertIn("keep", meta2.tags)
    
    def test_pretrain_record_complete(self):
        """Test PretrainRecord with full metadata"""
        record = PretrainRecord(
            raw="def hello_world(): print('Hello, World!')",
            meta=RecordMeta(
                tags=["code", "python", "function"],
                doc_language="en",
                extracted_by="pdf2seg v1.0",
                confidence=0.98,
                source_file="python_tutorial.pdf",
                notes="Clean function definition"
            )
        )
        
        self.assertIsNotNone(record.id)
        self.assertEqual(len(record.meta.tags), 3)
        self.assertEqual(record.meta.confidence, 0.98)
    
    def test_tokenized_input_schema(self):
        """Test TokenizedInput schema"""
        tokens = TokenizedInput(
            input=["def", " ", "hello", "(", ")", ":", " ", "print", "(", "'Hi'", ")"],
            tokenizer="oxbar",
            preserve_whitespace=True
        )
        
        self.assertTrue(tokens.preserve_whitespace)
        self.assertEqual(tokens.tokenizer, "oxbar")
        self.assertEqual(len(tokens.input), 11)
    
    def test_record_type_schema(self):
        """Test RecordType schema"""
        # Test all valid types with explicit literals
        rt_code = RecordType(type="code")
        self.assertEqual(rt_code.type, "code")
        
        rt_natural = RecordType(type="natural")
        self.assertEqual(rt_natural.type, "natural")
        
        rt_mixed = RecordType(type="mixed")
        self.assertEqual(rt_mixed.type, "mixed")
        
        # Test optional type (defaults to None)
        rt_none = RecordType()  # type: ignore
        self.assertIsNone(rt_none.type)
    
    def test_span_label_schema(self):
        """Test SpanLabel schema"""
        span = SpanLabel(
            span=(0, 2),
            label="noun_phrase",
            role="subject",
            text="the dog"
        )
        
        self.assertEqual(span.span, (0, 2))
        self.assertEqual(span.label, "noun_phrase")
        self.assertEqual(span.role, "subject")
    
    def test_type_literals_exist(self):
        """Test that all the new type literals are properly defined"""
        # Test that we can access the type literals
        self.assertTrue(hasattr(sys.modules[XPSpan.__module__], 'CodeLabel'))
        self.assertTrue(hasattr(sys.modules[XPSpan.__module__], 'CodeRole'))
        self.assertTrue(hasattr(sys.modules[XPSpan.__module__], 'NaturalLabel'))
        self.assertTrue(hasattr(sys.modules[XPSpan.__module__], 'NaturalRole'))
        self.assertTrue(hasattr(sys.modules[XPSpan.__module__], 'HybridRole'))
        self.assertTrue(hasattr(sys.modules[XPSpan.__module__], 'XBarRole'))
    
    def test_integration_full_record(self):
        """Test integration of all schema components"""
        # Create a complete record that uses multiple schema elements
        meta = RecordMeta(
            tags=["natural_language", "question"],
            doc_language="en",
            extracted_by="pdf2seg v1.0",
            confidence=0.87,
            source_file="linguistics_paper.pdf",
            notes="Contains interrogative structure"
        )
        
        record = PretrainRecord(
            raw="What is the syntax of this sentence?",
            meta=meta
        )
        
        # Validate the complete structure
        self.assertTrue(record.raw.endswith("?"))
        self.assertEqual(record.meta.doc_language, "en")
        # Handle Optional[float] type properly
        if record.meta.confidence is not None:
            self.assertGreater(record.meta.confidence, 0.8)
        self.assertIsNotNone(record.id)
        if record.id is not None:
            self.assertIsNotNone(record.id.id)


if __name__ == "__main__":
    unittest.main()
