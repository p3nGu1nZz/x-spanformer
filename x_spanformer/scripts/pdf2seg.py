#!/usr/bin/env python3

"""
Convert images into segmented text blocks using OCR and export to CSV.

Usage:
    python pdf2seg.py --input ./pages/ --output segments.csv
"""

import argparse
import csv
from pathlib import Path
import easyocr

def extract(input_path, out_path, lang='en', detail=0, paragraph=True):
    reader = easyocr.Reader([lang], gpu=True)
    input_path = Path(input_path)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    segments = []
    for image in image_files:
        results = reader.readtext(str(image), detail=detail, paragraph=paragraph)
        segments.extend([s.strip() for s in results if isinstance(s, str) and s.strip()])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for i, seg in enumerate(segments, 1):
            writer.writerow([i, seg])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Path to folder of image pages")
    parser.add_argument("-o", "--output", required=True, help="Path to output CSV file")
    parser.add_argument("-l", "--lang", default="en", help="Language(s) for OCR (e.g., en, ja)")
    parser.add_argument("-d", "--detail", type=int, choices=[0, 1], default=0, help="Detail level: 0 = text only, 1 = bounding boxes")
    args = parser.parse_args()
    extract(args.input, args.output, lang=args.lang, detail=args.detail, paragraph=args.paragraph)

if __name__ == "__main__":
    main()