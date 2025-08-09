"""
Annotation Validation Module

Validates annotations.jsonl files for:
- Positional accuracy (start/end positions match extracted text)
- Overlap rules (same start AND end not allowed, overlaps are OK)
- Data integrity (required fields, valid hierarchical levels)
- Span quality (proper linguistic boundaries)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in annotations."""
    issue_type: str
    sequence_number: int
    annotation_id: int
    description: str
    annotation: Optional[Dict] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    total_annotations: int
    valid_annotations: int
    issues: List[ValidationIssue]
    statistics: Dict[str, int]
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_annotations == 0:
            return 0.0
        return (len(self.issues) / self.total_annotations) * 100
    
    def get_issues_by_type(self) -> Dict[str, List[ValidationIssue]]:
        """Group issues by type."""
        grouped = defaultdict(list)
        for issue in self.issues:
            grouped[issue.issue_type].append(issue)
        return dict(grouped)


class AnnotationValidator:
    """Validates annotation files for correctness and quality."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.statistics = defaultdict(int)
    
    def validate_file(self, annotations_path: Path) -> ValidationReport:
        """
        Validate an annotations.jsonl file.
        
        Args:
            annotations_path: Path to annotations.jsonl file
            
        Returns:
            ValidationReport with all findings
        """
        self.issues.clear()
        self.statistics.clear()
        
        if not annotations_path.exists():
            logger.error(f"Annotations file not found: {annotations_path}")
            return ValidationReport(0, 0, [], {})
        
        annotations = self._load_annotations(annotations_path)
        if not annotations:
            return ValidationReport(0, 0, [], {})
        
        logger.info(f"Validating {len(annotations)} annotations...")
        
        # Run all validation checks
        self._validate_data_integrity(annotations)
        self._validate_positional_accuracy(annotations)
        self._validate_overlap_rules(annotations)
        self._validate_span_quality(annotations)
        self._calculate_statistics(annotations)
        
        valid_count = len(annotations) - len(self.issues)
        
        report = ValidationReport(
            total_annotations=len(annotations),
            valid_annotations=valid_count,
            issues=self.issues.copy(),
            statistics=dict(self.statistics)
        )
        
        self._log_summary(report)
        return report
    
    def _load_annotations(self, path: Path) -> List[Dict]:
        """Load annotations from JSONL file."""
        annotations = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        annotation = json.loads(line)
                        annotations.append(annotation)
                    except json.JSONDecodeError as e:
                        self.issues.append(ValidationIssue(
                            issue_type="json_decode_error",
                            sequence_number=0,
                            annotation_id=line_num,
                            description=f"Line {line_num}: Invalid JSON - {e}"
                        ))
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return []
        
        return annotations
    
    def _validate_data_integrity(self, annotations: List[Dict]) -> None:
        """Validate required fields and data types."""
        required_fields = {
            'id', 'sequence_number', 'raw', 'domain_type', 'start_pos', 
            'end_pos', 'xbar_label', 'text', 'source', 'hierarchical_level'
        }
        
        valid_levels = {'word_level', 'phrase_level', 'clause_level'}
        
        for ann in annotations:
            ann_id = ann.get('id', 'unknown')
            seq_num = ann.get('sequence_number', 0)
            
            # Check required fields
            missing_fields = required_fields - set(ann.keys())
            if missing_fields:
                self.issues.append(ValidationIssue(
                    issue_type="missing_fields",
                    sequence_number=seq_num,
                    annotation_id=ann_id,
                    description=f"Missing fields: {missing_fields}",
                    annotation=ann
                ))
                continue
            
            # Check hierarchical level
            if ann['hierarchical_level'] not in valid_levels:
                self.issues.append(ValidationIssue(
                    issue_type="invalid_level",
                    sequence_number=seq_num,
                    annotation_id=ann_id,
                    description=f"Invalid hierarchical_level: {ann['hierarchical_level']}",
                    annotation=ann
                ))
            
            # Check position types
            try:
                start_pos = int(ann['start_pos'])
                end_pos = int(ann['end_pos'])
                if start_pos < 0 or end_pos < 0 or start_pos >= end_pos:
                    self.issues.append(ValidationIssue(
                        issue_type="invalid_positions",
                        sequence_number=seq_num,
                        annotation_id=ann_id,
                        description=f"Invalid positions: start={start_pos}, end={end_pos}",
                        annotation=ann
                    ))
            except (ValueError, TypeError):
                self.issues.append(ValidationIssue(
                    issue_type="position_type_error",
                    sequence_number=seq_num,
                    annotation_id=ann_id,
                    description="start_pos/end_pos must be integers",
                    annotation=ann
                ))
    
    def _validate_positional_accuracy(self, annotations: List[Dict]) -> None:
        """Validate that extracted text matches positions in raw text."""
        for ann in annotations:
            if 'raw' not in ann or 'text' not in ann:
                continue
            
            try:
                start_pos = int(ann['start_pos'])
                end_pos = int(ann['end_pos'])
                raw_text = ann['raw']
                expected_text = ann['text']
                
                if end_pos > len(raw_text):
                    self.issues.append(ValidationIssue(
                        issue_type="position_out_of_bounds",
                        sequence_number=ann.get('sequence_number', 0),
                        annotation_id=ann.get('id', 'unknown'),
                        description=f"end_pos {end_pos} exceeds raw text length {len(raw_text)}",
                        annotation=ann
                    ))
                    continue
                
                actual_text = raw_text[start_pos:end_pos]
                
                if actual_text != expected_text:
                    self.issues.append(ValidationIssue(
                        issue_type="text_position_mismatch",
                        sequence_number=ann.get('sequence_number', 0),
                        annotation_id=ann.get('id', 'unknown'),
                        description=f"Expected '{expected_text}' but got '{actual_text}' at {start_pos}:{end_pos}",
                        annotation=ann
                    ))
            
            except (ValueError, TypeError, IndexError) as e:
                self.issues.append(ValidationIssue(
                    issue_type="position_validation_error",
                    sequence_number=ann.get('sequence_number', 0),
                    annotation_id=ann.get('id', 'unknown'),
                    description=f"Error validating positions: {e}",
                    annotation=ann
                ))
    
    def _validate_overlap_rules(self, annotations: List[Dict]) -> None:
        """
        Validate overlap rules:
        - Same start AND end positions not allowed (exact duplicates)
        - Overlapping spans with different start OR end are OK
        """
        # Group by sequence for efficient checking
        by_sequence = defaultdict(list)
        for ann in annotations:
            seq_num = ann.get('sequence_number', 0)
            by_sequence[seq_num].append(ann)
        
        for seq_num, seq_annotations in by_sequence.items():
            position_map: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
            
            # Group annotations by exact position
            for ann in seq_annotations:
                try:
                    start_pos = int(ann['start_pos'])
                    end_pos = int(ann['end_pos'])
                    position_map[(start_pos, end_pos)].append(ann)
                except (ValueError, TypeError):
                    continue  # Already handled in data integrity check
            
            # Check for exact duplicates (same start AND end)
            for (start_pos, end_pos), anns in position_map.items():
                if len(anns) > 1:
                    for ann in anns[1:]:  # First one is OK, rest are duplicates
                        self.issues.append(ValidationIssue(
                            issue_type="exact_duplicate",
                            sequence_number=seq_num,
                            annotation_id=ann.get('id', 'unknown'),
                            description=f"Exact duplicate at positions {start_pos}:{end_pos}",
                            annotation=ann
                        ))
    
    def _validate_span_quality(self, annotations: List[Dict]) -> None:
        """Validate span quality and linguistic boundaries."""
        for ann in annotations:
            text = ann.get('text', '')
            level = ann.get('hierarchical_level', '')
            
            # Check for empty or whitespace-only spans
            if not text or not text.strip():
                self.issues.append(ValidationIssue(
                    issue_type="empty_span",
                    sequence_number=ann.get('sequence_number', 0),
                    annotation_id=ann.get('id', 'unknown'),
                    description="Empty or whitespace-only span",
                    annotation=ann
                ))
                continue
            
            # Check for single character spans at higher levels
            if level in ['phrase_level', 'clause_level'] and len(text.strip()) == 1:
                # Valid single characters: punctuation, mathematical symbols, variables, alphanumeric
                # Only flag truly suspicious characters (whitespace, control chars, etc.)
                valid_single_chars = set('()[]{}~$%^&*+=<>|/\\.,;:!?"\'-_`@#')
                # Flag only if it's NOT in valid chars AND NOT alphanumeric AND NOT whitespace
                char = text.strip()
                if char and char not in valid_single_chars and not char.isalnum():
                    self.issues.append(ValidationIssue(
                        issue_type="suspicious_single_char",
                        sequence_number=ann.get('sequence_number', 0),
                        annotation_id=ann.get('id', 'unknown'),
                        description=f"Single non-standard character '{text}' at {level}",
                        annotation=ann
                    ))
            
            # Check for multi-word spans at word level (should be single words only)
            if level == 'word_level' and len(text.split()) > 1:
                self.issues.append(ValidationIssue(
                    issue_type="long_word_span",
                    sequence_number=ann.get('sequence_number', 0),
                    annotation_id=ann.get('id', 'unknown'),
                    description=f"Word-level span contains {len(text.split())} words: '{text[:50]}...'",
                    annotation=ann
                ))
    
    def _calculate_statistics(self, annotations: List[Dict]) -> None:
        """Calculate various statistics about the annotations."""
        self.statistics['total_annotations'] = len(annotations)
        
        # Count by hierarchical level
        for ann in annotations:
            level = ann.get('hierarchical_level', 'unknown')
            self.statistics[f'level_{level}'] = self.statistics.get(f'level_{level}', 0) + 1
        
        # Count by domain
        for ann in annotations:
            domain = ann.get('domain_type', 'unknown')
            self.statistics[f'domain_{domain}'] = self.statistics.get(f'domain_{domain}', 0) + 1
        
        # Count by source
        for ann in annotations:
            source = ann.get('source', 'unknown')
            self.statistics[f'source_{source}'] = self.statistics.get(f'source_{source}', 0) + 1
        
        # Count sequences
        unique_sequences = set(ann.get('sequence_number', 0) for ann in annotations)
        self.statistics['unique_sequences'] = len(unique_sequences)
        
        # Count issues by type
        issue_counts = defaultdict(int)
        for issue in self.issues:
            issue_counts[issue.issue_type] += 1
        
        for issue_type, count in issue_counts.items():
            self.statistics[f'issues_{issue_type}'] = count
    
    def _log_summary(self, report: ValidationReport) -> None:
        """Log validation summary."""
        logger.info("=" * 60)
        logger.info("ANNOTATION VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total annotations: {report.total_annotations}")
        logger.info(f"Valid annotations: {report.valid_annotations}")
        logger.info(f"Issues found: {len(report.issues)}")
        logger.info(f"Error rate: {report.error_rate:.2f}%")
        
        if report.issues:
            logger.info("\nIssues by type:")
            issue_counts = defaultdict(int)
            for issue in report.issues:
                issue_counts[issue.issue_type] += 1
            
            for issue_type, count in sorted(issue_counts.items()):
                logger.info(f"  {issue_type}: {count}")
        
        logger.info("\nLevel distribution:")
        for key, value in report.statistics.items():
            if key.startswith('level_'):
                level = key.replace('level_', '')
                logger.info(f"  {level}: {value}")
        
        logger.info("=" * 60)


def validate_annotations(annotations_path: str) -> ValidationReport:
    """
    Main validation function for annotations.
    
    Args:
        annotations_path: Path to annotations.jsonl file
        
    Returns:
        ValidationReport with comprehensive results
    """
    validator = AnnotationValidator()
    return validator.validate_file(Path(annotations_path))


def main():
    """Command line interface for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate X-Spanformer annotations")
    parser.add_argument("annotations_path", help="Path to annotations.jsonl file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    report = validate_annotations(args.annotations_path)
    
    # Print detailed issues if verbose
    if args.verbose and report.issues:
        print("\nDetailed Issues:")
        print("-" * 50)
        for issue in report.issues[:20]:  # Show first 20 issues
            print(f"{issue.issue_type}: Seq {issue.sequence_number}, "
                  f"ID {issue.annotation_id} - {issue.description}")
        
        if len(report.issues) > 20:
            print(f"... and {len(report.issues) - 20} more issues")


if __name__ == "__main__":
    main()
