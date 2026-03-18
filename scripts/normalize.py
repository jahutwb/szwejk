import re

def normalize_manuscript_text(md_text):
    # 1. Fix malformed Czech tags
    md_text = re.sub(r'\[\[]/cz]', '[[/cz]]', md_text)

    # 2. Merge broken lines inside paragraphs
    lines = md_text.splitlines()
    normalized = []
    buffer = ''
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Heading, scene break, or blank line: flush buffer
        if (stripped.startswith('#') or stripped.startswith('##') or
            stripped == '' or stripped == '***'):
            if buffer:
                normalized.append(buffer)
                buffer = ''
            normalized.append(stripped)
        else:
            if buffer:
                buffer += ' ' + stripped
            else:
                buffer = stripped
    if buffer:
        normalized.append(buffer)

    # 3. Remove isolated page numbers (lines with only digits)
    normalized = [l for l in normalized if not re.match(r'^\d{1,4}$', l.strip())]

    # 4. Remove obvious line fragments (single word lines)
    def is_fragment(line):
        return len(line.split()) == 1 and not line.startswith('#') and not line == '***'
    normalized = [l for l in normalized if not is_fragment(l)]

    # 5. Remove broken spacing inside words (very basic, e.g. 'a b c' -> 'abc' if all single chars)
    def fix_broken_words(line):
        return re.sub(r'(?<!\w)([a-zA-Z]) (?! )([a-zA-Z])(?!\w)', r'\1\2', line)
    normalized = [fix_broken_words(l) for l in normalized]

    return '\n'.join(normalized)
