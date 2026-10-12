"""Load a CV file (.docx or .pdf) into plain text for the LLM extractor.

Output format (.docx):
    # Section            <- Word "Heading 1" / "Title"
    ## Sub-section       <- "Heading 2"
    ### Skill group      <- "Heading 3"
    Entry header line    <- e.g. "Data Analyst | SnappShop | Tehran, Iran | 10/2023 – 02/2026"
    - bullet             <- Word list item or typed "•"
    Link text (URL)      <- hyperlinks keep their URL
"""

import re
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

TYPED_BULLET = re.compile(r"^[•▪◦○●\-–]\s+")
HEADING_PREFIX = {"title": "#", "heading 1": "#", "heading 2": "##", "heading 3": "###"}


def _clean_url(url: str) -> str:
    """Fix links damaged by copy-paste, e.g. '○%09https:/github.com/x' -> 'https://github.com/x'."""
    match = re.search(r"https?:/+.*", url)
    if not match:
        return url.strip()
    return re.sub(r"^(https?):/+", r"\1://", match.group(0)).strip()


def _paragraph_text(p: Paragraph) -> str:
    # runs + hyperlinks in document order; each link's URL goes right after its text
    parts: list[str] = []
    for item in p.iter_inner_content():
        if isinstance(item, Hyperlink) and item.url and item.text.strip():
            parts.append(f"{item.text} ({_clean_url(item.url)})")
        else:
            parts.append(item.text)  # blank/stray links are kept as plain text
    text = " ".join("".join(parts).split())  # collapse line breaks and double spaces
    if not text:
        return ""

    style = p.style.name.lower()
    if style in HEADING_PREFIX:
        return f"{HEADING_PREFIX[style]} {text}"

    is_list_item = (
        (p._p.pPr is not None and p._p.pPr.numPr is not None)  # bullets set on the paragraph
        or style.startswith("list")  # bullets set via a "List Bullet" style
    )
    if is_list_item or TYPED_BULLET.match(text):
        return f"- {TYPED_BULLET.sub('', text)}"
    return text


def _load_docx(path: Path) -> str:
    lines: list[str] = []
    for block in Document(str(path)).iter_inner_content():  # paragraphs + tables, in order
        if isinstance(block, Paragraph):
            lines.append(_paragraph_text(block))
        elif isinstance(block, Table):
            for row in block.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(line for line in lines if line)


def _load_pdf(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def load_cv_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    match path.suffix.lower():
        case ".docx":
            return _load_docx(path)
        case ".pdf":
            return _load_pdf(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix}")
