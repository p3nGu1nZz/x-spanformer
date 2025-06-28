We are building a tokenizer-free, span-aware encoder grounded in X-bar theory—called **X-Spanformer**. It learns to segment and fuse overlapping structural spans from raw token sequences (code, natural language, hybrids), powering symbolic routing and controller-aware representations.

Right now, We need help scaling up high-quality, interpretable training data for **Phase I span induction**.

I'm looking for collaborators to help generate:
- ✅ Single-token and multi-token span labels across `code`, `natural_language`, and `hybrid` domains
- ✅ Accurate `label`, `role`, and `text` fields aligned with each span
- ✅ Clear structure with nested units (think: call_exprs inside function bodies)

If you're into:
- ✨ Structured pretraining
- 🍄 Tokenizer-free segmentation
- 🌱 Linguistics-meets-AI modeling

...I'd love your help!

🧠 Tools provided (or in progress):
- JSON schema & validation scripts
- Span visualizer and entropy overlay maps
- Prototype dataset generation CLI

📬 DM me or drop a reply if you’re interested!  
Let’s build the first compositional span stack from the ground up 🧩

## **HOW TO CONTRIBUTE** 👉 https://github.com/p3nGu1nZz/x-spanformer/blob/main/CONTRIBUTING.md

# 🧭 Structural Fusion in X-Spanformer

This diagram visualizes how X-Spanformer extracts overlapping spans from token sequences (like `print('Hello')`), softly types them (e.g., `call_expr`, `literal`), and fuses them into a single controller vector `s`. That vector encodes the structure of the input and is injected into the transformer backbone via prefix tokens, attention bias, or gating pathways. During training, span selection is entropy-regularized and annealed over time, letting the model evolve from exploratory routing to confident, sparse structural control. Every fused signal influences how the model interprets and composes meaning—turning spans into steerable computation.

# 🌋 Span Density Map: Smushed Span Visualization in X-Spanformer

This 3D heatmap illustrates the stacked span structure over a tokenized input sequence. Each vertical “ridge” reflects the number of span labels covering that token—higher peaks indicate regions of dense structural overlap, such as nested modifiers, clause boundaries, or function calls like `print('Hello')`. As X-Spanformer learns to propose meaningful spans, these compositional hills emerge organically. During training, the entropy-annealed routing mechanism selectively sharpens these regions, guiding the controller vector `s` to focus on structurally rich spans while ignoring noise. The result is a soft structural blueprint that drives interpretable and efficient routing in downstream tasks.

# 📄 Example Training Record – Natural Language

```json
{
  "input": ["The", " ", "quick", " ", "brown", " ", "fox", " ", "jumps", " ", "over", " ", "the", " ", "lazy", " ", "dog", "."],
  "type": "natural_language",
  "span_labels": [
    {
      "span": [0, 0],
      "label": "determiner",
      "role": "noun specifier",
      "text": "The"
    },
    {
      "span": [2, 2],
      "label": "adjective",
      "role": "modifier",
      "text": "quick"
    },
    {
      "span": [4, 4],
      "label": "adjective",
      "role": "modifier",
      "text": "brown"
    },
    {
      "span": [6, 6],
      "label": "noun",
      "role": "subject",
      "text": "fox"
    },
    {
      "span": [8, 8],
      "label": "verb",
      "role": "predicate",
      "text": "jumps"
    },
    {
      "span": [10, 10],
      "label": "preposition",
      "role": "adverbial modifier",
      "text": "over"
    },
    {
      "span": [12, 12],
      "label": "determiner",
      "role": "noun specifier",
      "text": "the"
    },
    {
      "span": [14, 14],
      "label": "adjective",
      "role": "modifier",
      "text": "lazy"
    },
    {
      "span": [16, 16],
      "label": "noun",
      "role": "object",
      "text": "dog"
    },
    {
      "span": [17, 17],
      "label": "punctuation",
      "role": "terminator",
      "text": "."
    }
  ]
}
```

# 💬 Explanation

- Each word (plus space or punctuation) is tokenized as a separate entry in `input`.
- The `span_labels` tag each significant token with a structural `label` (noun, adjective, etc.) and functional `role` (subject, object, modifier).
- This format supports downstream training where spans are procedurally selected, and the model learns to recover both structure and role from unsegmented text.

# 💻 Example Training Record – Code

```json
{
  "input": ["let", " ", "x", " ", "=", " ", "42", ";"],
  "type": "code",
  "span_labels": [
    {
      "span": [0, 0],
      "label": "keyword",
      "role": "variable declaration",
      "text": "let"
    },
    {
      "span": [1, 1],
      "label": "space",
      "role": "separator",
      "text": " "
    },
    {
      "span": [2, 2],
      "label": "identifier",
      "role": "variable name",
      "text": "x"
    },
    {
      "span": [3, 3],
      "label": "space",
      "role": "separator",
      "text": " "
    },
    {
      "span": [4, 4],
      "label": "operator",
      "role": "assignment",
      "text": "="
    },
    {
      "span": [5, 5],
      "label": "space",
      "role": "separator",
      "text": " "
    },
    {
      "span": [6, 6],
      "label": "literal",
      "role": "numeric value",
      "text": "42"
    },
    {
      "span": [7, 7],
      "label": "delimiter",
      "role": "statement terminator",
      "text": ";"
    }
  ]
}
```

# 🧠 What's happening?

- Each token—identifiers, punctuation, spaces—is treated as a **literal structural unit**.
- The `span_labels` assign both a syntactic `label` (e.g., `keyword`, `literal`) and a functional `role` (e.g., `variable name`, `assignment`).
- This format allows X-Spanformer to learn **token-wise structure** and build **compositional span hypotheses** through entropy-annealed span selection and controller routing.

# 🧬 Example Training Record – Hybrid (Mixed Modality)

```json
{
  "input": ["To", " ", "define", " ", "a", " ", "constant", ",", " ", "use", " ", "`const", " ", "PI", " ", "=", " ", "3.14", "`", "."],
  "type": "mixed",
  "span_labels": [
    {
      "span": [0, 0],
      "label": "verb",
      "role": "instruction",
      "text": "To"
    },
    {
      "span": [2, 2],
      "label": "verb",
      "role": "instruction",
      "text": "define"
    },
    {
      "span": [6, 6],
      "label": "noun",
      "role": "object",
      "text": "constant"
    },
    {
      "span": [7, 7],
      "label": "punctuation",
      "role": "separator",
      "text": ","
    },
    {
      "span": [10, 10],
      "label": "verb",
      "role": "instruction",
      "text": "use"
    },
    {
      "span": [11, 11],
      "label": "code_delimiter",
      "role": "inline code open",
      "text": "`const"
    },
    {
      "span": [12, 12],
      "label": "identifier",
      "role": "constant name",
      "text": "PI"
    },
    {
      "span": [14, 14],
      "label": "operator",
      "role": "assignment",
      "text": "="
    },
    {
      "span": [16, 16],
      "label": "literal",
      "role": "float",
      "text": "3.14"
    },
    {
      "span": [17, 17],
      "label": "code_delimiter",
      "role": "inline code close",
      "text": "`"
    },
    {
      "span": [18, 18],
      "label": "punctuation",
      "role": "terminator",
      "text": "."
    }
  ]
}
```

# 🧠 Why this is important

- Mixed examples like this one teach X-Spanformer to handle embedded syntax shifts—e.g., inline code inside instructions.
- Inline code spans are bracketed with `code_delimiter`, and internal units like `PI`, `=`, `3.14` are labeled using code-appropriate types.
- Natural language verbs like `"define"` and `"use"` get captured in parallel, helping the model map structure and intention across modality boundaries.

# 🧬 Code Span Taxonomy: What It Captures and Why It Matters

This label set is designed to help X-Spanformer learn the underlying structure of programming languages—without relying on fixed tokens. It organizes span types into three main buckets:

---

# 1. **Syntactic Units**
These labels identify _what kind of thing_ a token is:

- `"keyword"` – Control words like `if`, `while`, `return`, `static`
- `"identifier"` – Named entities: variables, functions, classes
- `"operator"` – Symbols like `=`, `+`, `&&`, `->`
- `"delimiter"` – Braces, parentheses, brackets, semicolons
- `"literal"` – Strings, numbers, booleans, chars, regexes
- `"type"` – Data types: `int`, `float`, `char*`, custom typedefs
- `"specifier"` – Things like `const`, `static`, `volatile` (qualifiers)

---

# 2. **Structural + Semantic Roles**
Each label is paired with a role to explain what the token _does_:

- `"variable name"` for an identifier in a declaration
- `"assignment"` for an operator like `=`
- `"loop body"` or `"function body"` for `{ ... }` blocks
- `"function"` or `"macro invocation"` for named calls like `foo(...)`

These roles help the model understand scope, dependencies, and purpose—especially when labels repeat.

---

# 3. **Support and Surface Tokens**
These help with layout, doc parsing, and formatting sensitivity:

- `"space"` – Significant spacing (`"    "`, `"  "`) with role `"indent"` or `"separator"`
- `"newline"` – Line breaks separating logical units
- `"comment"` – Docstrings, line comments, or annotations
- `"preprocessor"` – C/C++ directives: `#include`, `#ifdef`, etc.
- `"call"`, `"block"`, `"control"` – Higher-level constructs like loops, `if` statements, and encapsulated logic

---

This schema lets X-Spanformer model the _structural spine_ of code—how parts relate and compose—without needing tokenizers or language-specific heuristics.

# ✍️ Natural Language Span Labels & Roles

This label set supports training X-Spanformer on `type: "natural_language"` sequences. Each token is tagged with a `label` (like `noun`, `verb`, etc.) and a context-sensitive `role`—allowing the model to learn syntax, compositional structure, and X-bar-style phrase projection without a tokenizer.

Here's how it's organized:

---

# 🧱 Core Lexical Categories

```json
{
  "noun": ["subject", "object", "complement", "agent", "theme"],
  "verb": ["predicate", "instruction", "tense anchor", "auxiliary"],
  "adjective": ["modifier", "attribute", "specifier"],
  "adverb": ["manner", "degree", "negation", "temporal"],
  "determiner": ["noun specifier", "quantifier", "article"],
  "preposition": ["adverbial modifier", "relation", "complement introducer"]
}
```

🟢 These tokens form the structural spine of the sentence: who did what, to whom, how, and with what specificity.

---

#### ✂️ Structural & Functional Tokens

```json
{
  "punctuation": ["terminator", "separator", "clause boundary"],
  "space": ["separator", "soft break", "phrase gap"],
  "newline": ["line break", "paragraph divider"]
}
```

🔘 These regulate phrasing, sentence boundaries, and formatting. They’re subtle but critical for structure induction.

---

#### 🌿 X-Bar Projections (Multi-token spans)

```json
{
  "xbar": [
    "X⁰ (head)",
    "X′ (intermediate)",
    "XP (phrase)",
    "specifier",
    "complement",
    "adjunct"
  ]
}
```

📐 These represent abstract constituents like noun phrases (NP), verb phrases (VP), and clause-level spans. They enable the model to learn hierarchical structure—essential for compositional reasoning.

# 🧠 Goal:
We want structured examples where every token (like a word, symbol, space, or newline) is stored in order—and some of those tokens are tagged with useful labels like `"noun"`, `"keyword"`, `"operator"`, along with the span index and role. These records teach the model how to discover structure in raw sequences.

---

# ✅ Step-by-Step Data Creation Guide

# 1️⃣ Choose an Input Sequence
Pick a short, clear sequence from one of these types:

- **Natural language**: _“The quick brown fox jumps over the lazy dog.”_
- **Code**: `for (int i = 0; i < 10; i++) { printf("Hello"); }`
- **Hybrid** (text + code): _“Use `return 0;` to exit a C function.”_

✳️ Keep it under ~20 tokens to start.

---

# 2️⃣ Tokenize It *Character-Consciously*
Split it into meaningful tokens, including spaces and punctuation.

For example:

```json
["for", " ", "(", "int", " ", "i", " ", "=", " ", "0", ";", ")"]
```

Each chunk—words, symbols, **spaces**, **tabs**, **newlines**—gets its own token. Why? Because spacing matters, especially in code or formatting-sensitive language.

---

# 3️⃣ Build the `input` Array
Put your tokens into the `"input"` field of your JSON object:

```json
"input": ["for", " ", "(", "int", " ", "i", " ", "=", " ", "0", ";", ")"]
```

---

# 4️⃣ Identify Key Spans
For each meaningful token or multi-token group:

- Mark its **start and end** index in the array
- Write a **`label`** (e.g. `"identifier"`, `"literal"`, `"noun"`, `"operator"`)
- Add a **`role`** (e.g. `"function name"`, `"assignment"`, `"subject"`)

📌 Use [this label + role reference](#) if you're unsure what to pick.

---

# 5️⃣ Create the `span_labels` Array
Each span is a dictionary like this:

```json
{
  "span": [3, 3],
  "label": "type",
  "role": "primitive",
  "text": "int"
}
```

All spans go into a `"span_labels"` list, sorted by their start index. For spans covering multiple tokens (e.g. `"printf(\"Hello\")"`), use a wider range like `[6, 10]`.

---

# 6️⃣ Combine Into a Single JSON Object
Here’s the whole structure:

```json
{
  "input": [...],
  "type": "code",  // or "natural_language" or "mixed"
  "span_labels": [
    { "span": [0, 0], "label": "keyword", "role": "loop type", "text": "for" },
    { "span": [3, 3], "label": "type", "role": "primitive", "text": "int" },
    { "span": [5, 5], "label": "identifier", "role": "loop variable", "text": "i" },
    ...
  ]
}
```

---

# 7️⃣ Optional: Add Multi-token Structural Spans
If you feel confident, add a few phrases or blocks:

```json
{ "span": [3, 5], "label": "xbar", "role": "XP (declarator)", "text": "int i" }
```

Don’t overdo it—just a few is enough. These help the model learn higher-level structure.

---

# 8️⃣ Test It With the Schema
Check:
- All spans are valid (no out-of-bound indices)
- `text` matches what's inside `input` between `span[0]` and `span[1]`
- No overlaps unless explicitly intended (like nested phrases)

I can help you automate this step with a validation script.

---

# 🔁 Repeat and Vary
Try creating:
- A code snippet
- A natural language sentence
- A Markdown + code mix
- A Python block with indentation
- A C macro or preprocessor directive

More diversity = better generalization.

