"""
Script to convert hybrid markdown manuscript to styled HTML for PDF/EPUB rendering.

- Input: hybrid manuscript markdown (with [[cz]]...[[/cz]] and [[vocab]]...[[/vocab]])
- Output: HTML file with embedded CSS, ready for WeasyPrint (PDF) or Pandoc (EPUB)

Usage:
    python render_hybrid_html.py --input manuscript.md --output manuscript.html
"""
import argparse
import re
from pathlib import Path

BOOK_CSS = '''
@font-face {
  font-family: 'Literata';
  src: url('assets/fonts/Literata/static/Literata-Regular.ttf') format('truetype');
  font-weight: normal;
  font-style: normal;
}
@font-face {
  font-family: 'Inter';
  src: url('assets/fonts/Inter/static/Inter_18pt-Regular.ttf') format('truetype');
  font-weight: normal;
  font-style: normal;
}
body {
  font-family: 'Literata', serif;
  font-size: 11pt;
  line-height: 1.42;
  margin-left: 20mm;
  margin-right: 15mm;
  margin-top: 18mm;
  margin-bottom: 25mm;
  text-align: justify;
  hyphens: auto;
}
p {
  text-indent: 1.2em;
  margin-top: 0;
  margin-bottom: 0.8em;
}
h1, h2, h3 {
  text-indent: 0;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
}
.czech {
  font-family: 'Inter', sans-serif;
  font-size: 0.95em;
  letter-spacing: 0.04em;
  color: #3a3a3a;
}
.vocab {
  font-weight: bold;
}
.footnotes {
  font-size: 0.9em;
  border-top: 1px solid #888;
  margin-top: 2em;
  padding-top: 1em;
}
.footnotes p {
  text-indent: 0;
  margin-bottom: 0.5em;
}
.page-break {
  page-break-after: always;
}
'''

def convert_markdown_to_html(md_text):
    # Replace [[cz]]...[[/cz]] with span
    html = re.sub(r'\[\[cz]](.*?)\[\[/cz]]', r'<span class="czech">\1</span>', md_text, flags=re.DOTALL)
    # Replace [[vocab]]...[[/vocab]] with span
    html = re.sub(r'\[\[vocab]](.*?)\[\[/vocab]]', r'<span class="vocab">\1</span>', html, flags=re.DOTALL)
    # Convert footnotes section
    html = re.sub(r'^---\s*Footnotes:', '<div class="footnotes"><h3>Footnotes</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\d+)\. (.+)$', r'<p>\1. \2</p>', html, flags=re.MULTILINE)
    html = html.replace('---', '</div>')
    # Convert paragraphs
    html = '\n'.join(f'<p>{line.strip()}</p>' if line.strip() else '' for line in html.splitlines())
    return html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    md_text = Path(args.input).read_text(encoding='utf-8')
    html_body = convert_markdown_to_html(md_text)
    html = f'''<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<title>Hybrid Book</title>
<style>{BOOK_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>'''
    Path(args.output).write_text(html, encoding='utf-8')

if __name__ == "__main__":
    main()
