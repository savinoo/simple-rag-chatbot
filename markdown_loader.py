"""Markdown loader with best-effort section/heading preservation.

Why:
- Section-level citations require knowing which heading a chunk came from.
- Many generic markdown loaders lose heading structure.

Strategy:
- Split markdown into segments by headings (#, ##, ### ...)
- Each segment becomes a Document with metadata.section_path
- The downstream text splitter can further chunk while preserving section_path
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


_heading_re = re.compile(r"^(#{1,6})\s+(.*)$")


def load_markdown_with_sections(path: str, source_name: Optional[str] = None) -> List[Document]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    source_name = source_name or p.name

    lines = text.splitlines()

    docs: List[Document] = []
    heading_stack: List[str] = []  # store heading titles
    buf: List[str] = []
    current_section_path: str = ""

    def flush():
        nonlocal buf, current_section_path
        content = "\n".join(buf).strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "section_path": current_section_path,
                    },
                )
            )
        buf = []

    for line in lines:
        m = _heading_re.match(line.strip())
        if m:
            # new section
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()

            # adjust stack
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_section_path = " > ".join(heading_stack)
            continue

        buf.append(line)

    flush()

    # If no headings at all, still return a single doc with empty section_path
    if not docs and text.strip():
        docs.append(Document(page_content=text, metadata={"source": source_name, "section_path": ""}))

    return docs
