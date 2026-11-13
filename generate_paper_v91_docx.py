# -*- coding: utf-8 -*-
"""
generate_paper_v91_docx.py
Converts paper_v91_FINAL.md into a publication-grade Word document (.docx)
matching Springer Nature (NCAA) editorial submission typography:
- Continuous line numbering (zero empty orphan numbered lines)
- Running header and footer page numbering
- Bold / bold-italic / italic / underline inline runs and backtick monospace code formatting
- Balanced-brace LaTeX-to-Unicode math translation with native Word sub/superscripts
- Publication three-line booktabs tables with proportional widths, repeating headers,
  cantSplit rows, and highlighted best (bold) & 2nd-best (underline)
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

_LBRACE = "\uE001"
_RBRACE = "\uE002"
_UNDER = "\uE003"
_TOK_START = "\uE010"
_TOK_END = "\uE011"


def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=55, bottom=55, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="10" w:space="0" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="10" w:space="0" w:color="000000"/>'
        f'<w:insideH w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def set_header_bottom_border(row):
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            f"</w:tcBorders>"
        )
        tcPr.append(tcBorders)


def set_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def set_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def set_table_fixed_layout(table, col_widths_inches):
    table.autofit = False
    tblPr = table._tbl.tblPr
    tblLayout = parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>')
    tblPr.append(tblLayout)
    tblW = parse_xml(
        f'<w:tblW {nsdecls("w")} w:w="{int(round(sum(col_widths_inches) * 1440))}" w:type="dxa"/>'
    )
    tblPr.append(tblW)
    for row in table.rows:
        for idx, width_in in enumerate(col_widths_inches):
            if idx < len(row.cells):
                row.cells[idx].width = Inches(width_in)


def add_continuous_line_numbers(doc):
    sectPr = doc.sections[0]._sectPr
    lnNumType = OxmlElement("w:lnNumType")
    lnNumType.set(qn("w:countBy"), "1")
    lnNumType.set(qn("w:restart"), "continuous")
    sectPr.append(lnNumType)


def add_header_footer(doc):
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    header = sec.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("PSO-TCWavKAN for Power Transformer Thermal Dynamics | Neural Computing and Applications")
    hrun.font.name = "Times New Roman"
    hrun.font.size = Pt(8.5)
    hrun.font.italic = True
    hrun.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fldSimple = OxmlElement("w:fldSimple")
    fldSimple.set(qn("w:instr"), "PAGE")
    fp._p.append(fldSimple)


def _extract_braced(s, brace_idx):
    """Given s[brace_idx] == '{', returns (inner_content, closing_brace_idx)."""
    depth = 0
    for j in range(brace_idx, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[brace_idx + 1 : j], j
    return s[brace_idx + 1 :], len(s) - 1


def _replace_macro_1arg(s, macro, fn):
    pattern = re.compile(re.escape(macro) + r"\s*\{")
    while True:
        m = pattern.search(s)
        if not m:
            break
        open_idx = m.end() - 1
        arg, close_idx = _extract_braced(s, open_idx)
        s = s[: m.start()] + fn(arg) + s[close_idx + 1 :]
    return s


def _replace_macro_2arg(s, macro, fn):
    pattern = re.compile(re.escape(macro) + r"\s*\{")
    while True:
        m = pattern.search(s)
        if not m:
            break
        open1 = m.end() - 1
        arg1, close1 = _extract_braced(s, open1)
        pos2 = close1 + 1
        while pos2 < len(s) and s[pos2].isspace():
            pos2 += 1
        if pos2 < len(s) and s[pos2] == "{":
            arg2, close2 = _extract_braced(s, pos2)
            s = s[: m.start()] + fn(arg1, arg2) + s[close2 + 1 :]
        else:
            s = s[: m.start()] + f"({arg1})" + s[close1 + 1 :]
    return s


def _flatten_nested_sub_sup(text):
    """Strips redundant braces/markers inside an already subscripted or superscripted slice."""
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^", "").replace("_", _UNDER)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _encode_sub_sup(s):
    """Turns _{...}/_x and ^{...}/^x into structured Word sub/superscript tokens."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ("_", "^") and i + 1 < n:
            kind = "SUB" if ch == "_" else "SUP"
            nxt = s[i + 1]
            if nxt == "{":
                inner, close_idx = _extract_braced(s, i + 1)
                clean_inner = _flatten_nested_sub_sup(inner)
                out.append(f"{_TOK_START}{kind}:{clean_inner}{_TOK_END}")
                i = close_idx + 1
                continue
            elif nxt.isalnum() or nxt in "+-*/()" or ord(nxt) > 127:
                out.append(f"{_TOK_START}{kind}:{nxt}{_TOK_END}")
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def latex_math_to_unicode(eq):
    """Translates LaTeX math cleanly into crisp Unicode + native Word sub/superscripts."""
    eq = eq.strip()

    # Protect escaped braces and underscores BEFORE stripping macros or sub/sup parsing
    eq = eq.replace(r"\{", _LBRACE).replace(r"\}", _RBRACE).replace(r"\_", _UNDER)
    eq = eq.replace(r"\|", "\u2016")

    # Convert degrees Celsius BEFORE stripping \mathrm/\text
    for deg_pat in [
        r"^\circ\mathrm{C}", r"^{\circ}\mathrm{C}", r"^{\circ}\text{C}",
        r"^\circ\text{C}", r"^\circ C", r"^{\circ}C", r"^\circ", r"^{\circ}",
    ]:
        eq = eq.replace(deg_pat, "\u00b0C" if deg_pat.endswith("C") or deg_pat.endswith("{C}") else "\u00b0")
    eq = eq.replace(r"\circ", "\u00b0")

    # Replace \left / \right delimiters BEFORE \le matching
    for delim in [
        r"\left(", r"\right)", r"\left[", r"\right]",
        r"\left|", r"\right|", r"\left.", r"\right.",
    ]:
        eq = eq.replace(delim, delim[-1] if delim[-1] != "." else "")
    eq = eq.replace(r"\lfloor", "\u230a").replace(r"\rfloor", "\u230b")
    eq = re.sub(r"\\(?:left|right)(?![a-zA-Z])", "", eq)

    # Balanced-brace macro unwrapping (looped to fixed point for nested fractions/accents)
    for _ in range(6):
        prev = eq
        for txt_macro in [r"\text", r"\mathrm", r"\mathbf", r"\boldsymbol", r"\operatorname"]:
            eq = _replace_macro_1arg(eq, txt_macro, lambda a: a)
        eq = _replace_macro_1arg(eq, r"\hat", lambda a: a + "\u0302")
        eq = _replace_macro_1arg(eq, r"\tilde", lambda a: a + "\u0303")
        eq = _replace_macro_1arg(eq, r"\bar", lambda a: a + "\u0304")
        eq = _replace_macro_1arg(eq, r"\sqrt", lambda a: "\u221a(" + a + ")")

        def _fmt_frac(a, b):
            a_s, b_s = a.strip(), b.strip()
            if len(a_s) <= 3 and len(b_s) <= 3 and a_s.isalnum() and b_s.isalnum():
                return f"{a_s}/{b_s}"
            return f"({a_s}) / ({b_s})"

        eq = _replace_macro_2arg(eq, r"\frac", _fmt_frac)
        if eq == prev:
            break

    # Blackboard bold & calligraphic fonts (Times New Roman / Cambria Math BMP-safe!)
    eq = eq.replace(r"\mathcal{O}", "O").replace(r"\mathcal{L}", "\u2112")
    eq = eq.replace(r"\mathcal{F}", "F").replace(r"\mathcal{K}", "K").replace(r"\mathcal{S}", "S")
    eq = eq.replace(r"\mathbb{R}", "\u211d").replace(r"\mathbb{C}", "\u2102")

    # Standard operators and Greek/math symbols (ordered longest-first to avoid prefix shadowing)
    repl = [
        (r"\rightarrow", "\u2192"),
        (r"\blacksquare", "\u25a0"),
        (r"\varepsilon", "\u03b5"),
        (r"\partial", "\u2202"),
        (r"\epsilon", "\u03b5"),
        (r"\approx", "\u2248"),
        (r"\subset", "\u2282"),
        (r"\lambda", "\u03bb"),
        (r"\equiv", "\u2261"),
        (r"\infty", "\u221e"),
        (r"\omega", "\u03c9"),
        (r"\sigma", "\u03c3"),
        (r"\gamma", "\u03b3"),
        (r"\alpha", "\u03b1"),
        (r"\times", "\u00d7"),
        (r"\qquad", "    "),
        (r"\Delta", "\u0394"),
        (r"\Theta", "\u0398"),
        (r"\Omega", "\u03a9"),
        (r"\dots", "\u2026"),
        (r"\quad", "  "),
        (r"\cdot", "\u00b7"),
        (r"\odot", "\u2299"),
        (r"\beta", "\u03b2"),
        (r"\chip", "\u03c7"),
        (r"\neq", "\u2260"),
        (r"\Phi", "\u03a6"),
        (r"\sum", "\u2211"),
        (r"\phi", "\u03c6"),
        (r"\psi", "\u03c8"),
        (r"\chi", "\u03c7"),
        (r"\tau", "\u03c4"),
        (r"\eta", "\u03b7"),
        (r"\cos", "cos"),
        (r"\exp", "exp"),
        (r"\log", "log"),
        (r"\lim", "lim"),
        (r"\max", "max"),
        (r"\pm", "\u00b1"),
        (r"\le", "\u2264"),
        (r"\ge", "\u2265"),
        (r"\ll", "\u226a"),
        (r"\to", "\u2192"),
        (r"\in", "\u2208"),
        (r"\mu", "\u03bc"),
        (r"\!", ""),
        (r"\;", " "),
        (r"\,", " "),
        ("---", "\u2014"),
        ("--", "\u2013"),
    ]
    for a, b in repl:
        eq = eq.replace(a, b)

    eq = re.sub(r"([a-zA-Z0-9)])(log|exp|cos|lim|max)\b", r"\1 \2", eq)

    # Encode subscripts and superscripts natively before restoring literal braces/underscores
    eq = _encode_sub_sup(eq)
    eq = eq.replace(_LBRACE, "{").replace(_RBRACE, "}").replace(_UNDER, "_")
    eq = re.sub(r"\s+", " ", eq).strip()
    return eq


def clean_inline_math_in_prose(text):
    """Converts $...$ segments cleanly for readable Word prose without literal dollar signs."""
    text = text.replace("---", " \u2014 ").replace("--", "\u2013")

    def _sub(m):
        return latex_math_to_unicode(m.group(1))

    return re.sub(r"\$([^$]+)\$", _sub, text)


def format_equation_text(raw_eq):
    eq = raw_eq.replace("$$", "").strip()
    eq_num_match = re.search(r"\(Eq\.\s*\d+\)", eq)
    eq_num = eq_num_match.group(0) if eq_num_match else ""
    if eq_num:
        eq = eq.replace(eq_num, "").strip()
    eq = latex_math_to_unicode(eq)
    if eq_num:
        eq = f"{eq}    {eq_num}"
    return eq


def _add_styled_run_with_subsup(paragraph, text, base_font, base_size, base_color,
                                is_bold=False, is_italic=False, is_underline=False):
    """Emits runs for plain text containing structured SUB/SUP tokens."""
    tok_pat = re.compile(rf"({_TOK_START}(?:SUP|SUB):[^{_TOK_END}]+{_TOK_END})")
    for part in tok_pat.split(text):
        if not part:
            continue
        if part.startswith(_TOK_START) and part.endswith(_TOK_END):
            inner = part[len(_TOK_START):-len(_TOK_END)]
            kind, val = inner.split(":", 1)
            r = paragraph.add_run(val)
            r.font.name = base_font
            r.font.size = Pt(base_size)
            r.font.bold = is_bold
            r.font.italic = is_italic
            r.font.underline = is_underline
            r.font.color.rgb = RGBColor(*base_color)
            if kind == "SUP":
                r.font.superscript = True
            else:
                r.font.subscript = True
        else:
            r = paragraph.add_run(part)
            r.font.name = base_font
            r.font.size = Pt(base_size)
            r.font.bold = is_bold
            r.font.italic = is_italic
            r.font.underline = is_underline
            r.font.color.rgb = RGBColor(*base_color)


def add_formatted_runs(paragraph, text, base_font="Times New Roman", base_size=11,
                       base_color=(0, 0, 0), is_bold=False, is_italic=False):
    text = clean_inline_math_in_prose(text)
    lines = re.split(r"<br\s*/?>", text)
    for line_idx, line in enumerate(lines):
        if line_idx > 0:
            paragraph.add_run("\n")
        tokens = re.split(
            r"(\*\*\*.*?\*\*\*|\*\*.*?\*\*|<u>.*?</u>|\*[^\s\*][^*]*?\*|`[^`]+`)",
            line,
        )
        for token in tokens:
            if not token:
                continue
            if token.startswith("***") and token.endswith("***") and len(token) >= 6:
                _add_styled_run_with_subsup(
                    paragraph, token[3:-3], base_font, base_size, base_color,
                    is_bold=True, is_italic=True,
                )
            elif token.startswith("**") and token.endswith("**") and len(token) >= 4:
                _add_styled_run_with_subsup(
                    paragraph, token[2:-2], base_font, base_size, base_color,
                    is_bold=True, is_italic=is_italic,
                )
            elif token.startswith("<u>") and token.endswith("</u>"):
                _add_styled_run_with_subsup(
                    paragraph, token[3:-4], base_font, base_size, base_color,
                    is_bold=is_bold, is_italic=is_italic, is_underline=True,
                )
            elif token.startswith("*") and token.endswith("*") and len(token) >= 2 and not token.startswith("**"):
                _add_styled_run_with_subsup(
                    paragraph, token[1:-1], base_font, base_size, base_color,
                    is_bold=is_bold, is_italic=True,
                )
            elif token.startswith("`") and token.endswith("`") and len(token) >= 2:
                _add_styled_run_with_subsup(
                    paragraph, token[1:-1], "Consolas", max(base_size - 1.0, 8.0), base_color,
                    is_bold=is_bold, is_italic=False,
                )
            else:
                _add_styled_run_with_subsup(
                    paragraph, token, base_font, base_size, base_color,
                    is_bold=is_bold, is_italic=is_italic,
                )


def _get_table_layout_profile(parsed_rows):
    """Returns (col_widths_inches_sum_6_5, font_size_pt, lr_pad_dxa) tailored per table."""
    num_cols = max(len(r) for r in parsed_rows)
    header = " | ".join(parsed_rows[0])
    if num_cols == 9 and "Seed MSEs" in header:  # Table 1
        return [0.46, 0.34, 0.34, 0.40, 0.90, 0.90, 1.56, 0.54, 1.06], 7.8, 45
    if num_cols == 8 and "DLinear" in header:  # Table 2
        return [0.46, 0.56, 0.88, 0.88, 0.92, 0.92, 0.94, 0.94], 7.5, 36
    if num_cols == 9 and "Paired Comparison" in header:  # Table 3
        return [1.62, 0.44, 0.44, 0.50, 0.56, 0.60, 0.62, 1.06, 0.66], 7.8, 45
    if num_cols == 5 and "Ablation Variant" in header:  # Table 4
        return [2.40, 1.15, 1.15, 0.85, 0.95], 8.5, 65
    if num_cols == 6 and "Total Params" in header:  # Table 5
        return [1.25, 1.35, 0.95, 1.05, 0.95, 0.95], 8.2, 55
    if num_cols == 6 and "Feature / Criterion" in header:  # Table 6
        return [1.10, 1.00, 1.02, 1.00, 1.10, 1.28], 8.0, 55
    # Generic fallback proportional to content width
    w = 6.50 / max(num_cols, 1)
    return [w] * num_cols, 8.2, 60


def build_manuscript():
    md_path = "paper_v91_FINAL.md"
    out_docx = "paper_v91_FINAL.docx"

    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    lines = md_text.split("\n")
    doc = docx.Document()
    doc.core_properties.author = "Phu Toan Nguyen"
    doc.core_properties.title = (
        "PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov-Arnold "
        "Networks for Power Transformer Thermal Dynamics Forecasting"
    )
    doc.core_properties.comments = "Neural Computing and Applications (Springer Nature) Submission"

    for sec in doc.sections:
        sec.top_margin = Inches(0.95)
        sec.bottom_margin = Inches(0.95)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    add_continuous_line_numbers(doc)
    add_header_footer(doc)

    i = 0
    total_lines = len(lines)
    after_block = False

    while i < total_lines:
        raw_line = lines[i]
        line = raw_line.strip()

        if not line or line == "---" or (line.startswith("<!--") and line.endswith("-->")):
            i += 1
            continue

        if line.startswith("# "):
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(10)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, title_text, base_size=15.0, base_color=(0, 32, 96), is_bold=True)
            after_block = False
            i += 1
            continue

        if line.startswith("**Running Title:**"):
            rt_text = line.replace("**Running Title:**", "").strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(6)
            add_formatted_runs(p, f"**Running Title:** {rt_text}", base_size=10, is_italic=True)
            after_block = False
            i += 1
            continue

        if line.startswith("**Author:**") or line.startswith("**Affiliation:**") or line.startswith("**Email:**") or line.startswith("**ORCID:**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(3)
            add_formatted_runs(p, line, base_size=10.5)
            after_block = False
            i += 1
            continue

        if line.startswith("## "):
            h1_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True
            add_formatted_runs(p, h1_text, base_size=13, base_color=(0, 32, 96), is_bold=True)
            after_block = False
            i += 1
            continue

        if line.startswith("### "):
            h2_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(9)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            add_formatted_runs(p, h2_text, base_size=11.5, is_bold=True)
            after_block = False
            i += 1
            continue

        if line.startswith("#### "):
            h3_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(7)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            add_formatted_runs(p, h3_text, base_size=11, is_bold=True, is_italic=True)
            after_block = False
            i += 1
            continue

        if line.startswith("**Keywords:**"):
            kw_text = line.replace("**Keywords:**", "").strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(10)
            r_lbl = p.add_run("Keywords: ")
            r_lbl.font.name = "Times New Roman"
            r_lbl.font.size = Pt(10.5)
            r_lbl.font.bold = True
            add_formatted_runs(p, kw_text, base_size=10.5, is_italic=True)
            after_block = False
            i += 1
            continue

        if line.startswith("**Table "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            add_formatted_runs(p, line, base_size=10.0)
            after_block = False
            i += 1
            continue

        if line.startswith("*Note:"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(9)
            note_clean = line[1:-1] if (line.endswith("*") and line.count("*") == 2) else line
            add_formatted_runs(p, note_clean, base_size=9.0, is_italic=True)
            after_block = False
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < total_lines and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            parsed_rows = []
            for tl in table_lines:
                if re.match(r"^\|[\s\-:]+(\|[\s\-:]+)+\|$", tl):
                    continue
                raw_cells = tl.split("|")[1:-1]
                parsed_rows.append([c.strip() for c in raw_cells])

            if parsed_rows:
                num_rows = len(parsed_rows)
                num_cols = max(len(r) for r in parsed_rows)
                col_widths, cell_pt, lr_pad = _get_table_layout_profile(parsed_rows)

                tbl = doc.add_table(rows=num_rows, cols=num_cols)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(tbl)
                set_table_fixed_layout(tbl, col_widths)

                for r_idx, row_data in enumerate(parsed_rows):
                    row = tbl.rows[r_idx]
                    set_row_cant_split(row)
                    is_header = (r_idx == 0)
                    if is_header:
                        set_header_bottom_border(row)
                        set_row_header(row)
                    for c_idx in range(num_cols):
                        cell = row.cells[c_idx]
                        cell_text = row_data[c_idx] if c_idx < len(row_data) else ""
                        set_cell_margins(cell, top=50, bottom=50, left=lr_pad, right=lr_pad)
                        if is_header:
                            set_cell_shading(cell, "F2F4F8")
                        cp = cell.paragraphs[0]
                        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER if (is_header or c_idx > 0) else WD_ALIGN_PARAGRAPH.LEFT
                        cp.paragraph_format.space_before = Pt(1)
                        cp.paragraph_format.space_after = Pt(1)
                        cp.paragraph_format.line_spacing = 1.0
                        add_formatted_runs(cp, cell_text, base_size=cell_pt, is_bold=is_header)

                after_block = True
            continue

        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < total_lines and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\r\n"))
                i += 1
            if i < total_lines and lines[i].strip().startswith("```"):
                i += 1

            if code_lines:
                tbl = doc.add_table(rows=1, cols=1)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_fixed_layout(tbl, [6.50])
                set_row_cant_split(tbl.rows[0])
                cell = tbl.rows[0].cells[0]
                set_cell_margins(cell, top=90, bottom=90, left=130, right=130)
                set_cell_shading(cell, "F8FAFC")
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(
                    parse_xml(
                        f'<w:tcBorders {nsdecls("w")}>'
                        f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="1A365D"/>'
                        f'  <w:left w:val="none"/>'
                        f'  <w:bottom w:val="single" w:sz="12" w:space="0" w:color="1A365D"/>'
                        f'  <w:right w:val="none"/>'
                        f"</w:tcBorders>"
                    )
                )
                first_p = cell.paragraphs[0]
                for c_idx, cl in enumerate(code_lines):
                    p = first_p if c_idx == 0 else cell.add_paragraph()
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.05
                    run = p.add_run(cl)
                    run.font.name = "Consolas"
                    run.font.size = Pt(8.0)
                    if cl.startswith("Algorithm ") or cl.startswith("Require") or cl.startswith("Ensure"):
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
                after_block = True
            continue

        if line.startswith("![Fig"):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                fig_caption = m.group(1).strip()
                fig_path = m.group(2).strip()
                if os.path.exists(fig_path):
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.paragraph_format.space_before = Pt(7 if not after_block else 9)
                    p_img.paragraph_format.space_after = Pt(3)
                    p_img.paragraph_format.keep_with_next = True
                    p_img.add_run().add_picture(fig_path, width=Inches(5.65))

                next_idx = i + 1
                while next_idx < total_lines and not lines[next_idx].strip():
                    next_idx += 1
                if next_idx < total_lines and lines[next_idx].strip().startswith("**Fig"):
                    cap_text = lines[next_idx].strip()
                    i = next_idx
                else:
                    cap_text = fig_caption

                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p_cap.paragraph_format.space_before = Pt(2)
                p_cap.paragraph_format.space_after = Pt(8)
                add_formatted_runs(p_cap, cap_text, base_size=9.5)
            after_block = False
            i += 1
            continue

        if (line.startswith("$$") and line.endswith("$$")) or re.search(r"\(Eq\.\s*\d+\)\s*$", line):
            clean_eq = format_equation_text(line)
            p_eq = doc.add_paragraph()
            p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_eq.paragraph_format.space_before = Pt(6 if after_block else 5)
            p_eq.paragraph_format.space_after = Pt(5)
            p_eq.paragraph_format.line_spacing = 1.12
            _add_styled_run_with_subsup(p_eq, clean_eq, "Times New Roman", 10.5, (0, 0, 0))
            after_block = False
            i += 1
            continue

        if line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            p_b = doc.add_paragraph(style="List Bullet")
            p_b.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_b.paragraph_format.space_before = Pt(5 if after_block else 1)
            p_b.paragraph_format.space_after = Pt(2.5)
            p_b.paragraph_format.line_spacing = 1.12
            add_formatted_runs(p_b, bullet_text, base_size=11)
            after_block = False
            i += 1
            continue

        if re.match(r"^\d+\.\s+", line):
            num_text = line
            p_n = doc.add_paragraph()
            p_n.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_n.paragraph_format.left_indent = Inches(0.25)
            p_n.paragraph_format.space_before = Pt(5 if after_block else 1)
            p_n.paragraph_format.space_after = Pt(2.5)
            p_n.paragraph_format.line_spacing = 1.12
            add_formatted_runs(p_n, num_text, base_size=11)
            after_block = False
            i += 1
            continue

        if re.match(r"^\[\d+\]\s+", line):
            p_ref = doc.add_paragraph()
            p_ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_ref.paragraph_format.space_before = Pt(0.5)
            p_ref.paragraph_format.space_after = Pt(2.2)
            p_ref.paragraph_format.line_spacing = 1.08
            p_ref.paragraph_format.left_indent = Inches(0.35)
            p_ref.paragraph_format.first_line_indent = Inches(-0.35)
            add_formatted_runs(p_ref, line, base_size=9.5)
            after_block = False
            i += 1
            continue

        p_body = doc.add_paragraph()
        p_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_body.paragraph_format.space_before = Pt(6 if after_block else 0)
        p_body.paragraph_format.space_after = Pt(5)
        p_body.paragraph_format.line_spacing = 1.12
        add_formatted_runs(p_body, line, base_size=11)
        after_block = False
        i += 1

    doc.save(out_docx)
    print(f"Successfully generated {out_docx}")


if __name__ == "__main__":
    build_manuscript()
