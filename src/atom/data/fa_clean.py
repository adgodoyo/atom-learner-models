"""HTML/MathML cleaning for FoundationalASSIST problem bodies.

Vendored (lightly adapted) from the dataset's own shipped cleaner,
`Datasets/FoundationalAssist/Code/clean_utils.py` (Worden et al. 2026, CC-BY-NC).
The dataset's shipped cleaning code is reused rather than reimplemented; it is copied
into the repo so a fresh clone is self-contained (the raw DATA never enters the repo —
only this cleaning CODE does). The cleaned text feeds the semantic codec.

Preserves: inline MathML (fractions/super/subscripts), Wiris math images (from the
`data-mathml` attribute), table structure (| separators), image placeholders.
"""

from __future__ import annotations

import html

import pandas as pd
from bs4 import BeautifulSoup, NavigableString


def parse_mathml_element(elem):
    """Recursively render a MathML element to plain text."""
    if elem.name is None:
        return str(elem).strip()

    if elem.name == "mfrac":
        children = [c for c in elem.children if c.name]
        if len(children) >= 2:
            num = parse_mathml_element(children[0])
            denom = parse_mathml_element(children[1])
            return f"({num}/{denom})"
        return elem.get_text(strip=True)

    elif elem.name == "msup":
        children = [c for c in elem.children if c.name]
        if len(children) >= 2:
            base = parse_mathml_element(children[0])
            exp = parse_mathml_element(children[1])
            return f"{base}^{exp}"
        return elem.get_text(strip=True)

    elif elem.name == "msub":
        children = [c for c in elem.children if c.name]
        if len(children) >= 2:
            base = parse_mathml_element(children[0])
            sub = parse_mathml_element(children[1])
            return f"{base}_{sub}"
        return elem.get_text(strip=True)

    elif elem.name == "msqrt":
        content = parse_mathml_element_children(elem)
        return f"√({content})"

    elif elem.name == "mo":
        op = elem.get_text(strip=True)
        if op in ["÷", "×", "·", "+", "-", "=", "<", ">", "≤", "≥", "≠"]:
            return f" {op} "
        return op

    elif elem.name in ["mn", "mi", "mtext"]:
        return elem.get_text(strip=True)

    elif elem.name in ["mrow", "math", "mpadded", "mstyle"]:
        return parse_mathml_element_children(elem)

    else:
        return elem.get_text(strip=True)


def parse_mathml_element_children(elem):
    """Render all children of a MathML element."""
    parts = []
    for child in elem.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
        elif child.name:
            parts.append(parse_mathml_element(child))
    return "".join(parts)


def clean_problem_body(text):
    """Clean an HTML problem body into natural text with MathML handled.

    - inline MathML (<math>, <mfrac>, <msup>, ...) -> (4/3), x^2
    - Wiris math images (data-mathml attribute)     -> [15 ÷ 12]
    - tables                                         -> [Table: Col1 | Col2 ...]
    - other images                                   -> [Image: alt] / [image]
    - HTML entities decoded; whitespace collapsed
    """
    if pd.isna(text) or text == "":
        return ""
    soup = BeautifulSoup(str(text), "html.parser")

    # 1. Inline MathML
    for math in soup.find_all("math"):
        parsed = parse_mathml_element(math)
        math.replace_with(f" {parsed} ")

    # 2. Wiris (and other) images
    for img in soup.find_all("img"):
        alt = img.get("alt", "")
        src = img.get("src", "")
        data_mathml = img.get("data-mathml", "")

        if "wiris" in src.lower() or "pluginwiris" in src:
            if alt and alt.strip() and alt not in ["NO ALT", "NONE"]:
                img.replace_with(f" [{alt.strip()}] ")
            elif data_mathml:
                math_str = data_mathml.replace("«", "<").replace("»", ">").replace("¨", '"')
                msoup = BeautifulSoup(math_str, "html.parser")
                math_elem = msoup.find("math")
                if math_elem:
                    mtext = parse_mathml_element(math_elem)
                else:
                    mtext = msoup.get_text(separator="")
                mtext = mtext.replace("§#247;", "÷").replace("§#215;", "×")
                mtext = mtext.replace("§#8722;", "-").replace("§#160;", " ").replace("§#183;", "·")
                mtext = mtext.replace("§#", "&#")
                mtext = html.unescape(mtext).strip()
                img.replace_with(f" [{mtext}] " if mtext else " [math] ")
            else:
                img.replace_with(" [math] ")
        elif alt and alt.strip():
            img.replace_with(f" [Image: {alt.strip()[:100]}] ")
        else:
            img.replace_with(" [image] ")

    # 3. Tables
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with(f"\n[Table:\n{chr(10).join(rows)}]\n")
        else:
            table.decompose()

    text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = " ".join(text.split())
    return text.strip()
