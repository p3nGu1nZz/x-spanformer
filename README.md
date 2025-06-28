# 🧠 X-Spanformer

**Tokenizer-free, span-aware transformer grounded in X-bar theory**  
X-Spanformer learns to segment and fuse overlapping spans directly from raw input—code, natural language, or hybrid strings—without relying on tokenizers. It uses span-induced controller vectors to steer computation and model structure in a linguistically grounded, modality-flexible way.

---

## 🚀 Key Features

- **Tokenizer-Free Encoding** – no subword splits or external segmenters  
- **Span-Aware Attention Routing** – structure is learned and fused into prefix/control signals  
- **Multi-Domain Compositionality** – supports code, prose, REPLs, Markdown, etc.  
- **Entropy-Annealed Training** – spans are initially exploratory, then crystallize over time  
- **X-Bar Inspired Representation** – spans learned hierarchically, with linguistic roles and projections  

---

## 📦 Data Format

Training examples follow this schema:

```json
{
  "input": ["The", " ", "quick", " ", "brown", " ", "fox", "."],
  "type": "natural_language",
  "span_labels": [
    { "span": [0, 0], "label": "determiner", "role": "specifier", "text": "The" },
    { "span": [2, 4], "label": "adjective_phrase", "role": "modifier", "text": "quick brown" },
    { "span": [6, 6], "label": "noun", "role": "subject", "text": "fox" }
  ]
}
```

For full details, see [`/examples`](./examples) and the companion compiler agent [`ox-bar`](https://github.com/.../ox-bar).

---

## 🧪 Data Preprocessing

To generate semantically coherent pretraining data without tokenizers, we use the [`pdf2seg`](https://pypi.org/project/pdf2seg) package:

```bash
pip install pdf2seg
```

Process scanned or structured PDFs into entropy-minimized text spans:

```bash
pdf2seg -i input.pdf -o raw_spans/
```

Use the output as either raw training strings (for unsupervised Phase I) or compile with `oxbar` to produce labeled span records.

This enables X-Spanformer to bootstrap span boundaries from real-world documents with high structural signal, without relying on brittle tokenization.

---

## 🧰 Repository Structure

```
x-spanformer/
├── model/                # Core encoder + span fusion modules
├── dataset/              # Record loader, validation, and augmentation
├── train/                # Training config, Lightning loop, curriculum schedules
├── visuals/              # Span density maps, entropy overlays, structure debugging
├── experiments/          # Prototype tests: controller variants, dropout, fusion modes
├── docs/                 # Concept diagrams, architecture notes, citations
└── examples/             # Code / NL / hybrid span records
```

---

## 🧪 Training Utilities

- `span-validator.py` — check span bounds, role-label pairings, schema conformity  
- `entropy-map.py` — visualize smushed spans and structural attention regions  
- `role-index.json` — map roles to IDs for supervised fusion routing  
- `xbar-guide.md` — taxonomy for natural language, code, and hybrid inputs  

---

## 🔧 Compiler Agents

### [`oxbar`](https://github.com/.../ox-bar)

Generate structured span-labeled records using local LLMs:

```bash
oxbar compile input.txt --type mixed --output spans.json
```

Supports retry logic, confidence scoring, and mode switching.

### [`pdf2seg`](https://pypi.org/project/pdf2seg)

Segment PDF documents into structured clauses using OCR + spaCy:

```bash
pdf2seg -i paper.pdf -o spans/
```

Ideal for extracting domain-specific clause boundaries from scientific papers, REPL transcripts, or code-heavy PDFs.

---

## 🧬 Architectural Inspiration

- **Linguistics:** X-bar phrase theory, projection-based syntax, span recursion  
- **Biomimicry:** Mycelial routing, compositional inference, entropy-driven adaptation  
- **Transformer Augmentation:** Span-aware attention, controller modulation, dropout-driven routing  

---

## 🤝 Contributing

We welcome span explorers, linguistically curious devs, and tokenizer skeptics.

Ways to help:
- Label new examples using `oxbar` or manual annotations  
- Extend the span role taxonomy for underrepresented domains (e.g., REPLs, math, RST)  
- Build new controller fusion heads or injection pathways  
- Analyze span induction across language families, treebanks, or doc formats  
- Visualize structural routing dynamics in longer sequences

Start with [`CONTRIBUTING.md`](./CONTRIBUTING.md) to onboard.

---

## 📄 Citation & License

This research and code are licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

```
Copyright (c) 2025  
TAU SYSTEMS by NAXZYU CORP.
```

### 📚 Zenodo Preprint  

**[https://zenodo.org/records/15750962](https://zenodo.org/records/15750962)**

```bibtex
@misc{rawson2025xspanformer,
  title        = {X-Spanformer: Tokenizer-Free Span Induction with Structural Fusion},
  author       = {Rawson, Kara and Chrzanowski, Aimee},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15750962},
  url          = {https://zenodo.org/records/15750962}
}
```
