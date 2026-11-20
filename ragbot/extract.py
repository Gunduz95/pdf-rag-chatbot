"""Step 1: PDF -> plain text (one '--- Page N ---' header per page)."""
from __future__ import annotations

from pathlib import Path

from . import config


def pages_to_text(pages: list[str]) -> str:
    return "\n".join(f"--- Page {i} ---\n{text}" for i, text in enumerate(pages, start=1))


def extract_pdf(pdf_path: Path = config.PDF_PATH, txt_path: Path = config.TXT_PATH) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = pages_to_text([page.extract_text() or "" for page in reader.pages])
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    print(f"{len(reader.pages)} pages -> {len(text)} characters saved to {txt_path}")
    return text
