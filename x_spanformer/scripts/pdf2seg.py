#!/usr/bin/env python3
import argparse, csv, random, hashlib, json
from pathlib import Path
from PIL import Image
import numpy as np, easyocr, spacy
from pdf2image import convert_from_path
from spacy.cli.download import download
from spacy.util import is_package
from rich.console import Console
from rich.progress import track
from rich.text import Text

DPI, MIN, MAX, LM, LANG, GPU = 300, 1, 5, "en_core_web_sm", ['en'], True
C = Console()

load = lambda m=LM: (
    spacy.load(m) if is_package(m)
    else (C.print(Text(f"[boot] Installing {m}", style="yellow"))
    or download(m) or spacy.load(m))
)

def cut(txt, nlp, a=MIN, b=MAX):
    s = [x.text.strip() for x in nlp(txt).sents if x.text.strip()]
    out, i = [], 0
    while i < len(s):
        k = random.randint(a, b)
        out.append(" ".join(s[i:i+k]))
        i += k
    return out

def hash(p): return hashlib.sha256(Path(p).name.encode()).hexdigest()[:8]

def state(dir): 
    f = dir / f"{dir.name}.json"
    if not f.exists(): return {"ocr": []}
    try:
        d = json.loads(f.read_text("utf-8"))
        return d if isinstance(d, dict) else {"ocr": []}
    except: return {"ocr": []}

def save(dir, d): (dir / f"{dir.name}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

def pdf(p, d, dpi=DPI):
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{d.name}.json"
    meta = state(d)
    img = list(d.glob(f"{d.name}-p*.png"))
    if meta.get("rendered") and len(img) == meta.get("pages", 0): return sorted(img), meta
    pages = convert_from_path(str(p), dpi=dpi)
    meta["pages"] = len(pages)
    for i, im in enumerate(track(pages, description="📄 Render")):
        fn = d / f"{d.name}-p{i:03}.png"
        if not fn.exists(): im.save(fn, "PNG")
    meta["rendered"] = True
    save(d, meta)
    return sorted(d.glob(f"{d.name}-p*.png")), meta

def ocr(imgs, meta, d):
    txts, R = [], easyocr.Reader(LANG, gpu=GPU)
    for i, fn in enumerate(track(imgs, description="🔍 OCR")):
        pid = f"{d.name}-p{i:03}"
        txt = d / f"{pid}.txt"
        if i in meta.get("ocr", []) and txt.exists(): txts.append(txt.read_text("utf-8")); continue
        arr = np.array(Image.open(fn))
        lines = [x.strip() for x in R.readtext(arr, detail=0, paragraph=True) if isinstance(x, str) and x.strip()]
        txt.write_text("\n".join(lines), encoding="utf-8")
        meta.setdefault("ocr", []).append(i)
        save(d, meta)
        txts.append("\n".join(lines))
    return txts

def filt(spans, nlp, min_len=24, min_words=4):
    keep = []
    for s in spans:
        doc = nlp(s)
        if len(s.strip()) < min_len: continue
        if sum(t.is_alpha for t in doc) < min_words:
            if not any(c.isnumeric() or c in "=/^*%$" for c in s): continue
        keep.append(s)
    return keep

def save_csv(segs, out, src):
    out = Path(out)
    name = hash(src)
    csvfile = out / f"{name}.csv" if out.is_dir() else out
    csvfile.parent.mkdir(parents=True, exist_ok=True)
    C.print(Text(f"💾 Writing {len(segs)} spans → {csvfile}", style="blue"))
    with open(csvfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "text"])
        for i, s in enumerate(track(segs, description="✨ Saving..."), 1): w.writerow([i, s])
    C.print(Text("✅ Done", style="magenta"))

def run(inp, out):
    name = hash(inp)
    d = Path(out) / name
    imgs, meta = pdf(inp, d)
    raw = ocr(imgs, meta, d)
    spans = [x for txt in raw for x in cut(txt, nlp)]
    spans = filt(spans, nlp)
    save_csv(spans, out, inp)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    global nlp
    nlp = load()
    run(args.input, args.output)

if __name__ == "__main__": main()