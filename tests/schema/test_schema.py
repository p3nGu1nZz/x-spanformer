import unittest
from pydantic import ValidationError

from x_spanformer.schema import (
    identifier, metadata, pretrain_record, scoring, source, span, status,
    tokenization, typing, validation, xbar
)


class TestSchemas(unittest.TestCase):
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
        self.assertIsNone(
            metadata.RecordMeta(
                extracted_by="test",
                source_file="test.txt",
                notes="test note",
                doc_language="en",
                confidence=None,
            ).confidence
        )

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
