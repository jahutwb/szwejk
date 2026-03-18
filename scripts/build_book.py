from scripts.normalize import normalize_manuscript_text
from scripts.structure import detect_structure
"""
Book body and glossary generator for hybrid Polish–Czech novel.
Reads manuscript.md and injects content into template placeholders.
No layout, font, or preamble logic is generated here.
"""
import re
import sys
from pathlib import Path
import subprocess

TEMPLATE_PATH = Path(__file__).parent.parent / "template" / "template.tex"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_TEX = OUTPUT_DIR / "book.tex"

def escape_latex(text):
    """Escape LaTeX special characters."""
    return (text.replace('\\', r'\textbackslash{}')
                .replace('{', r'\{')
                .replace('}', r'\}')
                .replace('%', r'\%')
                .replace('$', r'\$')
                .replace('_', r'\_')
                .replace('&', r'\&')
                .replace('#', r'\#'))

def parse_manuscript(md_text):
    seen_vocab = set()
    glossary = []  # list of (text, gloss)
    lines = md_text.splitlines()
    latex_lines = []
    chapter_pat = re.compile(r'^# (.+)$')
    section_pat = re.compile(r'^## (.+)$')
    scenebreak_pat = re.compile(r'^\*\*\*$')
    cznote_pat = re.compile(r'\[\[cznote:([^|\]]+)\|([^\]]+)\]\]')
    cz_pat = re.compile(r'\[\[cz\]\](.*?)\[\[/cz\]\]', re.DOTALL)

    def replace_cznote(m):
        text, gloss = m.group(1), m.group(2)
        if text not in seen_vocab:
            seen_vocab.add(text)
            glossary.append((text, gloss))
            return f'\\cznote{{{escape_latex(text)}}}{{{escape_latex(gloss)}}}'
        else:
            return f'\\czplain{{{escape_latex(text)}}}'

    def replace_cz(m):
        text = m.group(1)
        return f'\\czplain{{{escape_latex(text)}}}'

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            latex_lines.append("")
            continue
        # Headings
        m = chapter_pat.match(line)
        if m:
            latex_lines.append(f'\\chapter{{{escape_latex(m.group(1))}}}')
            continue
        m = section_pat.match(line)
        if m:
            latex_lines.append(f'\\section*{{{escape_latex(m.group(1))}}}')
            continue
        # Scene break
        if scenebreak_pat.match(line):
            latex_lines.append('\\scenebreak')
            continue
        # Inline markup
        line = cznote_pat.sub(replace_cznote, line)
        line = cz_pat.sub(replace_cz, line)
        latex_lines.append(line)
    return "\n".join(latex_lines), glossary

def make_glossary(glossary):
    if not glossary:
        return ""
    lines = ["\\begin{description}"]
    for text, gloss in glossary:
        lines.append(f'  \\item[\\textbf{{{escape_latex(text)}}}] {escape_latex(gloss)}')
    lines.append("\\end{description}")
    return "\n".join(lines)

def build_book(md_path):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()
    norm_text = normalize_manuscript_text(md_text)
    # Write normalized text for inspection
    with open(OUTPUT_DIR / "normalized.txt", "w", encoding="utf-8") as f:
        f.write(norm_text)

    # Validation: no malformed tags
    if '[[' in norm_text and not re.search(r'\[\[(cz|cznote):?|/cz\]\]', norm_text):
        print("ERROR: Malformed [[cz tags remain after normalization.")
        sys.exit(1)

    # Structure detection
    latex_lines, headings = detect_structure(norm_text)
    # Validation: at least one heading
    if not headings:
        print("ERROR: No valid headings detected in manuscript.")
        sys.exit(1)
    # Validation: first 500 chars not truncated
    if len(norm_text[:500]) > 0 and norm_text[:500].endswith((' ', '-', '.', ',', ':', ';')):
        pass
    elif len(norm_text[:500]) > 0 and not norm_text[:500].endswith((' ', '-', '.', ',', ':', ';')):
        print("ERROR: First 500 characters end in a truncated fragment.")
        sys.exit(1)

    # Write headings for debug
    with open(OUTPUT_DIR / "headings.txt", "w", encoding="utf-8") as f:
        for h in headings:
            f.write(h + '\n')

    # Only after structure detection, process Czech markup
    latex_body, glossary = parse_manuscript('\n'.join(latex_lines))
    glossary_content = make_glossary(glossary)

    # Optionally extract title/author/subtitle from the first lines or leave blank
    book_title = ""
    book_author = ""
    book_subtitle = ""
    # Optionally: parse for metadata at the top of the manuscript

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    latex_full = (template
        .replace("%%BOOK_TITLE%%", book_title)
        .replace("%%BOOK_AUTHOR%%", book_author)
        .replace("%%BOOK_SUBTITLE%%", book_subtitle)
        .replace("%%BOOK_CONTENT%%", latex_body)
        .replace("%%GLOSSARY_CONTENT%%", glossary_content)
        .replace("%%META_CONTENT%%", "")
    )
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex_full)
    # Compile PDF
    try:
        subprocess.run([
            "latexmk",
            "-lualatex",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-output-directory=output",
            "output/book.tex"
        ], cwd=OUTPUT_DIR.parent, check=True)
    except Exception as e:
        print("PDF compilation failed:", e)
        print("You may need to install latexmk and LuaLaTeX, and ensure fonts are available.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/build_book.py manuscript.md")
        sys.exit(1)
    build_book(sys.argv[1])
