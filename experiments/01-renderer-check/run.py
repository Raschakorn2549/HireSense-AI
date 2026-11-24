"""Experiment 01 — renderer check.

Questions this script answers (see docs/02-dataset-strategy.md §6.1, §6.2):

  1. When Edge (via Playwright) prints a TWO-COLUMN resume to PDF, in what order
     do PDF text extractors read the text back? Column by column, or line by line
     across both columns (which mixes sidebar and main content)?
  2. Does "invisible" text survive into the PDF's text layer? If Chromium drops it
     at print time, an adversarial test built on it would pass without testing
     anything.
  3. Can we make an image-only PDF (for the OCR fallback test) with Edge + Pillow,
     so that reportlab is not needed as a dependency?

Run from the project root:
    .venv/Scripts/python.exe -I -X utf8 experiments/01-renderer-check/run.py

This is an experiment, not production code. Output goes next to this file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber
import pymupdf  # AGPL-3.0 — experiment only, see results
import pypdfium2 as pdfium
from PIL import Image
from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Test documents
# ---------------------------------------------------------------------------

# Every marker below must appear EXACTLY ONCE in the visible text, and only in
# its own column. The script asserts this before trusting any measurement.
LEFT_MARKERS = [
    "CONTACT", "Bangkok, Thailand", "SKILLS", "PostgreSQL", "Docker",
    "Kubernetes", "EDUCATION", "B.Sc. Computer Science", "Example University",
]
RIGHT_MARKERS = [
    "SUMMARY", "three years of experience", "EXPERIENCE", "Acme Example Co",
    "two million requests per day", "PROJECTS", "Invoice parser",
    "reduced manual data entry",
]
SECTION_HEADERS = ["CONTACT", "SKILLS", "EDUCATION", "SUMMARY", "EXPERIENCE", "PROJECTS"]

TWO_COLUMN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  @page { size: A4; margin: 16mm; }
  body { font-family: Arial, sans-serif; font-size: 10.5pt; color: #222; margin: 0; }
  header { border-bottom: 2px solid #333; padding-bottom: 6px; margin-bottom: 10px; }
  header h1 { margin: 0; font-size: 20pt; }
  .grid { display: grid; grid-template-columns: 32% 1fr; gap: 18px; }
  aside { background: #f2f4f7; padding: 10px; }
  h2 { font-size: 11pt; letter-spacing: 1px; margin: 12px 0 4px; }
  ul { margin: 0; padding-left: 16px; }
  p { margin: 0 0 6px; }
</style></head><body>
<header>
  <h1>Somchai Example <small>(he/him)</small></h1>
  <div>somchai.e@example.com · 080-000-0000</div>
</header>
<div class="grid">
  <aside>
    <h2>CONTACT</h2>
    <p>Bangkok, Thailand</p>
    <h2>SKILLS</h2>
    <ul><li>Python</li><li>FastAPI</li><li>PostgreSQL</li><li>Docker</li><li>Kubernetes</li></ul>
    <h2>EDUCATION</h2>
    <p>B.Sc. Computer Science</p>
    <p>Example University, 2022</p>
  </aside>
  <main>
    <h2>SUMMARY</h2>
    <p>Backend developer with three years of experience designing and operating
       HTTP services. Comfortable owning a service from schema design to on-call.</p>
    <h2>EXPERIENCE</h2>
    <p><b>Backend Developer</b> — Acme Example Co, 2023-01 to Present</p>
    <ul>
      <li>Built and maintained a public API handling two million requests per day.</li>
      <li>Moved the deployment pipeline to containers and cut release time in half.</li>
      <li>Mentored two junior engineers through their first production incidents.</li>
    </ul>
    <h2>PROJECTS</h2>
    <p><b>Invoice parser</b> — an internal tool that reads supplier invoices and
       reduced manual data entry for the finance team.</p>
  </main>
</div>
</body></html>"""

# Each technique hides its own unique token, so we can tell which ones survive.
HIDDEN_TECHNIQUES = {
    "white_on_white": '<span style="color:#ffffff">ZQWHITE Python Kubernetes AWS</span>',
    "transparent":    '<span style="color:transparent">ZQTRANSPARENT Python Kubernetes AWS</span>',
    "opacity_zero":   '<span style="opacity:0">ZQOPACITY Python Kubernetes AWS</span>',
    "font_size_zero": '<span style="font-size:0">ZQSIZEZERO Python Kubernetes AWS</span>',
    "font_size_1px":  '<span style="font-size:1px">ZQSIZEONE Python Kubernetes AWS</span>',
    "off_page":       '<span style="position:absolute;left:-3000px;top:40px">ZQOFFPAGE Python Kubernetes AWS</span>',
    "display_none":   '<span style="display:none">ZQDISPLAYNONE Python Kubernetes AWS</span>',  # control
}

HIDDEN_HTML = (
    '<!doctype html><html><head><meta charset="utf-8"><style>'
    "@page{size:A4;margin:16mm} body{font-family:Arial,sans-serif;background:#fff}"
    "</style></head><body>"
    "<h1>Visible Applicant</h1><p>Visible skills: Excel, Communication.</p>"
    + "".join(f"<p>{snippet}</p>" for snippet in HIDDEN_TECHNIQUES.values())
    + "</body></html>"
)

# ---------------------------------------------------------------------------
# Extractors — three independent engines
# ---------------------------------------------------------------------------

def extract_pdfplumber(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_pypdfium2(path: Path) -> str:
    doc = pdfium.PdfDocument(str(path))
    try:
        return "\n".join(page.get_textpage().get_text_range() for page in doc)
    finally:
        doc.close()


def extract_pymupdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


EXTRACTORS = {
    "pdfplumber (pdfminer)": extract_pdfplumber,
    "pypdfium2 (PDFium)": extract_pypdfium2,
    "pymupdf (MuPDF)": extract_pymupdf,
}

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def visible_text_of(html: str) -> str:
    """Rough visible text of the test HTML, used only to validate the markers."""
    text = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", text)


def validate_markers() -> None:
    visible = visible_text_of(TWO_COLUMN_HTML)
    for marker in LEFT_MARKERS + RIGHT_MARKERS:
        count = visible.count(marker)
        assert count == 1, f"marker {marker!r} appears {count} times — measurement would be ambiguous"


WORD = re.compile(r"[A-Za-z][A-Za-z.+\-]*")


def column_vocabularies() -> tuple[set[str], set[str]]:
    """Words that occur in only one column of the source HTML.

    v1 of this script classified lines by matching whole marker PHRASES. That
    under-counted mixed lines: a phrase broken by a line wrap matched nothing,
    so a line like "Kubernetes day." was not recognised as mixing both columns.
    Classifying word by word is robust to wrapping.
    """
    left_html = re.search(r"<aside>(.*?)</aside>", TWO_COLUMN_HTML, re.S).group(1)
    right_html = re.search(r"<main>(.*?)</main>", TWO_COLUMN_HTML, re.S).group(1)
    left = {w.lower() for w in WORD.findall(visible_text_of(left_html))}
    right = {w.lower() for w in WORD.findall(visible_text_of(right_html))}
    return left - right, right - left


LEFT_ONLY, RIGHT_ONLY = column_vocabularies()


def column_order_report(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    labels: list[str] = []
    mixed_lines: list[str] = []
    for line in lines:
        words = {w.lower() for w in WORD.findall(line)}
        has_left = bool(words & LEFT_ONLY)
        has_right = bool(words & RIGHT_ONLY)
        if has_left and has_right:
            labels.append("MIX")
            mixed_lines.append(line)
        elif has_left:
            labels.append("L")
        elif has_right:
            labels.append("R")

    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    positions = {h: text.find(h) for h in SECTION_HEADERS}
    header_order = [h for h, pos in sorted(positions.items(), key=lambda kv: kv[1]) if pos >= 0]
    # Join wrapped lines before checking, so a phrase split by a line break is
    # not reported as missing text (v1 reported a false "missing" for exactly this).
    flat = re.sub(r"\s+", " ", text)
    missing = [m for m in LEFT_MARKERS + RIGHT_MARKERS if m not in flat]
    return {
        "header_order": header_order,
        "label_sequence": "".join({"L": "L", "R": "R", "MIX": "x"}[x] for x in labels),
        "column_switches": switches,
        "mixed_lines": mixed_lines,
        "missing_markers": missing,
    }


def hidden_text_report(path: Path) -> dict:
    tokens = {name: re.search(r"ZQ[A-Z]+", snippet).group(0) for name, snippet in HIDDEN_TECHNIQUES.items()}
    result: dict = {}
    for extractor_name, extract in EXTRACTORS.items():
        text = extract(path)
        result[extractor_name] = {name: (token in text) for name, token in tokens.items()}

    # What a defence could look at: per-character colour and size from pdfplumber.
    char_info: dict = {}
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        chars = page.chars
        stream = "".join(c["text"] for c in chars)
        for name, token in tokens.items():
            idx = stream.find(token)
            if idx < 0:
                continue
            c = chars[idx]
            char_info[name] = {
                "size": round(c["size"], 2),
                "fill_colour": c.get("non_stroking_color"),
                "x0": round(c["x0"], 1),
                "page_width": round(page.width, 1),
            }
    return {"found": result, "first_char_of_token": char_info}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    validate_markers()
    two_col_pdf = OUT_DIR / "two-column.pdf"
    hidden_pdf = OUT_DIR / "hidden-text.pdf"
    png = OUT_DIR / "two-column.png"
    image_pdf = OUT_DIR / "image-only.pdf"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        edge_version = browser.version
        page = browser.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=2)

        page.set_content(TWO_COLUMN_HTML)
        page.pdf(path=str(two_col_pdf), format="A4", print_background=True)
        page.screenshot(path=str(png), full_page=True)

        page.set_content(HIDDEN_HTML)
        page.pdf(path=str(hidden_pdf), format="A4", print_background=True)
        browser.close()

    # Image-only PDF with Pillow — no reportlab.
    Image.open(png).convert("RGB").save(image_pdf, "PDF", resolution=192.0)

    results = {
        "edge_version": edge_version,
        "two_column": {name: column_order_report(fn(two_col_pdf)) for name, fn in EXTRACTORS.items()},
        "hidden_text": hidden_text_report(hidden_pdf),
        "image_only": {
            name: len(fn(image_pdf).strip()) for name, fn in EXTRACTORS.items()
        },
        "file_sizes_bytes": {p.name: p.stat().st_size for p in (two_col_pdf, hidden_pdf, image_pdf)},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, fn in EXTRACTORS.items():
        safe = name.split()[0]
        (OUT_DIR / f"two-column.{safe}.txt").write_text(fn(two_col_pdf), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
