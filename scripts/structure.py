import re

def detect_structure(norm_text):
    """
    Detect Svejk-style headings and emit LaTeX structure.
    Returns (latex_lines, headings_list)
    """
    lines = norm_text.splitlines()
    latex_lines = []
    headings = []
    part_pat = re.compile(r"^D[IÍ]L\s*\d+['´]?\s+(.+)$", re.IGNORECASE)
    chapter_pat = re.compile(r"^\d+['´]?\s+(.+)$")
    intro_pat = re.compile(r"^ÚVOD$", re.IGNORECASE)
    section_pat = re.compile(r"^\((\d+)\)$")
    in_intro = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if intro_pat.match(stripped):
            latex_lines.append(r"\chapter*{Úvod}")
            latex_lines.append(r"\addcontentsline{toc}{chapter}{Úvod}")
            headings.append("Úvod")
            in_intro = True
            continue
        m = part_pat.match(stripped)
        if m:
            title = m.group(1).capitalize()
            latex_lines.append(f"\\part{{{title}}}")
            headings.append(f"PART: {title}")
            continue
        m = chapter_pat.match(stripped)
        if m:
            title = m.group(1).capitalize()
            latex_lines.append(f"\\chapter{{{title}}}")
            headings.append(f"CHAPTER: {title}")
            continue
        m = section_pat.match(stripped)
        if m:
            num = m.group(1)
            latex_lines.append(f"\\section*{{{num}}}")
            headings.append(f"SECTION: {num}")
            continue
        # preserve blank lines
        if stripped == '':
            latex_lines.append('')
            continue
        # normal paragraph
        latex_lines.append(stripped)
    return latex_lines, headings
