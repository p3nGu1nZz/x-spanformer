# 🤝 Contributing to X-Spanformer

We welcome collaborators interested in tokenizer-free modeling, span induction, structural learning, and sustainable AI tooling. Whether you’re a linguist, engineer, researcher, or just tokenizer-curious—we’d love to have you onboard.

---

## 🧠 What You Can Help With

### 📦 Span Dataset Generation
- Label natural language, code, or hybrid inputs using [ox-bar](https://github.com/.../ox-bar)
- Expand span coverage (multi-token spans, nested structures)
- Add examples with diverse formatting (REPLs, Markdown, preprocessor macros)

### 🧪 Model + Experimentation
- Test new fusion strategies (controller bias, span gating, dropout)
- Evaluate span density maps + entropy patterns across domains
- Develop benchmarks for span quality or generalization

### 🛠 Tools & Infrastructure
- Improve the `ox-bar` compiler or validator
- Add new critic agents or retry loops
- Build visualization tools (SVG overlays, inspector GUIs)

---

## 🧰 Getting Started

1. **Fork this repo**  
2. **Create a virtual environment**  
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   ```
3. **Run the test suite to validate your environment**
   ```bash
   python -m pytest tests/ -v
   ```
4. **Explore** the architecture and data processing pipelines
   - Study the pipelines in `x_spanformer/pipelines/`
   - Examine the documentation in `docs/`
   - Review the LaTeX paper in `docs/paper/`

---

## 🧪 Development Workflow

Before pushing changes:

```bash
# Run the complete test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/pipelines/     # Pipeline tests
python -m pytest tests/embedding/     # Embedding module tests
python -m pytest tests/schema/        # Schema validation tests

# Run tests with coverage
python -m pytest tests/ -v --cov=x_spanformer
```

### Code Quality
- Follow existing code style and patterns
- Add tests for new functionality  
- Update documentation for architectural changes
- Ensure mathematical correctness aligns with the paper

---

## 📝 Style Guidelines

- Use meaningful commit messages (e.g., `add XP span for noun phrase`, not `fix stuff`)
- Keep span record contributions modular (per-file, ≤ 100 samples)
- Match the taxonomy definitions in `/taxonomy` whenever possible

---

## 📩 Submitting a PR

1. Open a pull request from your fork
2. Describe your changes and what domain(s) it affects
3. Add tests or examples if relevant
4. We’ll review, validate, and discuss edge cases or style tweaks if needed

---

## 📄 Licensing + Attribution

All contributions will be licensed under **CC-BY 4.0** and associated with the project’s authorship. If you'd like explicit contributor credit in the documentation or paper, let us know.
