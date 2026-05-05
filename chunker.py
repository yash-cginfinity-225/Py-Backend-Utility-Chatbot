"""
chunker.py  — Semantic Chunker v6  (heading/page-scoped)
---------------------------------------------------------
Produces semantically coherent chunks from Azure Document Intelligence
Markdown output.

Chunking strategy (v6)
=======================
1. PRIMARY boundary  : headings  (H1–H5)
2. SECONDARY boundary: page overflow — if accumulated content within a single
   heading section exceeds CHUNK_MAX_TOKENS, it is split at paragraph /
   sentence boundaries WITH overlap.  Tables and figures that straddle the
   split boundary are kept whole in the LATER chunk (never cut in half).
3. Tables and figures are NEVER emitted as separate chunks.  They are
   appended inline to the heading section they belong to, and the containing
   chunk's flags (has_table, is_table, table_count, …) are set accordingly.
4. Rich metadata is preserved: topic/subtopic/section breadcrumb, page range,
   content-type flags, counts, doc-level metadata.

NEW in v6
=========
- `region` field added — fixed value "New York" for all chunks.
- `energy_utility_name` field added — derived from the immediate parent
  folder name of the source .md file (e.g. documents/ConEd/file.md → "ConEd").

Performance notes (v6 inherits v5 optimisations)
==================================================
- All regexes compiled at module level.
- Tag detection uses str.startswith() / == before any regex.
- Cached sanitised stem in _make_id.
- chunk_directory uses ProcessPoolExecutor for parallel file processing.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_MAX_TOKENS     = 600   # soft ceiling for a single chunk
CHUNK_OVERLAP_TOKENS = 80    # overlap carried into next split-piece
CHARS_PER_TOKEN      = 4

FIXED_REGION = "New York"

PROCESS_TABLE_LABELS: set[str] = {
    "PROCESS NUMBER", "PROCESS NAME", "PROCESS DEFINITION",
    "TRIGGER(S)", "PROCESS INPUTS", "PROCESS OUTPUTS",
    "SUB OR PRECEDING PROCESSES", "PROCESS RULES", "COMMENTS", "COMMENT",
}

# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled regexes
# ─────────────────────────────────────────────────────────────────────────────

_RE_PAGE_NUM       = re.compile(r'PageNumber["\s=]+(\d+)')
_RE_PAGE_HDR_VAL   = re.compile(r'PageHeader="([^"]+)"')
_RE_SUBMITTED      = re.compile(r'Submitted:\s*([^"]+)')
_RE_HEADING        = re.compile(r'^(#{1,5})\s+(.*)')
_RE_HRULE          = re.compile(r'^-{3,}$')

_RE_TR             = re.compile(r'<tr[^>]*>(.*?)</tr>',  re.DOTALL | re.IGNORECASE)
_RE_TD             = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
_RE_TAG_STRIP      = re.compile(r'<[^>]+>')
_RE_WHITESPACE     = re.compile(r'\s+')

_RE_FIG_DESC       = re.compile(r'<!--\s*AI Description:(.*?)-->', re.DOTALL)
_RE_FIG_CAP        = re.compile(r'<figcaption>(.*?)</figcaption>', re.DOTALL | re.IGNORECASE)
_RE_HTML_CMT       = re.compile(r'<!--.*?-->', re.DOTALL)
_RE_SPECIAL_CHARS  = re.compile(r'[☒✓□]')

_RE_UNSAFE_KEY     = re.compile(r'[^A-Za-z0-9_\-=]+')
_RE_LIST_ITEM      = re.compile(r'^\s*(\d+\.|[-•*])\s', re.MULTILINE)

_RE_PDF_LINK       = re.compile(r'Source PDF[^[\n]*\[([^\]]+)\]\(([^)\n]+)\)', re.IGNORECASE)
_RE_PDF_PLAIN      = re.compile(r'Source PDF[^h\n]*(https?://[^\s)\n]+)', re.IGNORECASE)
_RE_TOTAL_PAGES    = re.compile(r'Total Pages[^0-9\n]*(\d+)', re.IGNORECASE)
_RE_TOTAL_TABLES   = re.compile(r'Total Tables[^0-9\n]*(\d+)', re.IGNORECASE)
_RE_VERSION        = re.compile(r'VERSION\s+([\d.]+)', re.IGNORECASE)
_RE_SUBMITTED_META = re.compile(r'Submitted:\s*([^\n"*]+)', re.IGNORECASE)
_RE_CASE_NUM       = re.compile(r'Case\s+([\d-]+[A-Z]-[\d]+)', re.IGNORECASE)

_STEM_CACHE: dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id:        str
    source:          str
    chunk_index:     int

    # ── Location ──────────────────────────────────────────────────────────────
    page_start:      Optional[int]
    page_end:        Optional[int]

    # ── Hierarchy ─────────────────────────────────────────────────────────────
    heading_level:   int
    topic:           str
    subtopic:        str
    section_title:   str
    section:         str

    # ── Source document reference ─────────────────────────────────────────────
    source_pdf_url:  str
    source_pdf_name: str

    # ── NEW in v6: utility / geography ────────────────────────────────────────
    region:              str = FIXED_REGION   # always "New York"
    energy_utility_name: str = ""             # parent folder name of the .md file

    # ── Content ───────────────────────────────────────────────────────────────
    content:         str  = ""
    content_type:    str  = "text"   # "text" | "table" | "figure" | "list" | "mixed"

    # ── Flags (defaults → must be last) ──────────────────────────────────────
    has_table:    bool = False
    has_figure:   bool = False
    is_table:     bool = False   # v5+: always False — tables are never standalone
    is_figure:    bool = False   # v5+: always False — figures are never standalone
    table_count:  int  = 0
    figure_count: int  = 0

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_id(source: str, index: int, content: str) -> str:
    h = hashlib.md5(f"{source}:{index}:{content[:80]}".encode()).hexdigest()[:12]
    if source not in _STEM_CACHE:
        stem = Path(source).stem
        stem = _RE_UNSAFE_KEY.sub('_', stem).strip('_')
        _STEM_CACHE[source] = stem
    return f"{_STEM_CACHE[source]}_{index:04d}_{h}"


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _detect_content_type(text: str) -> str:
    has_table  = "|" in text
    has_figure = (
        "[Figure" in text or
        "[Flowchart" in text or
        "[Process Steps]" in text
    )
    if has_table and has_figure:
        return "mixed"
    if has_table:
        return "table"
    if has_figure:
        return "figure"
    if _RE_LIST_ITEM.search(text):
        return "list"
    return "text"


def _split_text_with_overlap(text: str) -> list[str]:
    """Split a long plain-text string respecting paragraph / sentence boundaries."""
    max_chars     = CHUNK_MAX_TOKENS     * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            for sep in ("\n\n", ". ", "\n"):
                pos = text.rfind(sep, start, end)
                if pos > start + overlap_chars:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end - overlap_chars
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# HTML table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_html_table(html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for rm in _RE_TR.finditer(html):
        cells = _RE_TD.findall(rm.group(1))
        cleaned = [_RE_WHITESPACE.sub(' ', _RE_TAG_STRIP.sub('', c)).strip()
                   for c in cells]
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _is_process_table(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    first_col = {r[0].upper().rstrip(':') for r in rows if r}
    return bool(first_col & PROCESS_TABLE_LABELS)


def _process_table_to_text(rows: list[list[str]]) -> str:
    fields: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2:
            key = row[0].rstrip(':').strip()
            val = " ".join(row[1:]).strip()
            if key and val:
                fields[key] = val
        elif len(row) == 1 and row[0] and fields:
            last = list(fields)[-1]
            fields[last] += " " + row[0]
    priority = [
        "PROCESS NUMBER", "PROCESS NAME", "PROCESS DEFINITION",
        "TRIGGER(S)", "PROCESS RULES", "PROCESS INPUTS", "PROCESS OUTPUTS",
        "SUB OR PRECEDING PROCESSES", "COMMENTS", "COMMENT",
    ]
    seen: set[str] = set()
    lines: list[str] = []
    for k in priority:
        if k in fields:
            lines.append(f"{k}: {fields[k]}")
            seen.add(k)
    for k, v in fields.items():
        if k not in seen:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _generic_table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(r) for r in rows)


# ─────────────────────────────────────────────────────────────────────────────
# Figure helper
# ─────────────────────────────────────────────────────────────────────────────

def _figure_to_text(html: str) -> str:
    desc_m  = _RE_FIG_DESC.search(html)
    ai_desc = _RE_WHITESPACE.sub(' ', desc_m.group(1)).strip() if desc_m else ""

    cap_m   = _RE_FIG_CAP.search(html)
    caption = _RE_WHITESPACE.sub(' ', cap_m.group(1)).strip() if cap_m else ""

    plain = _RE_HTML_CMT.sub('', html)
    plain = _RE_TAG_STRIP.sub('', plain)
    plain = _RE_SPECIAL_CHARS.sub('', plain)
    plain = _RE_WHITESPACE.sub(' ', plain).strip()

    parts: list[str] = []
    if caption:
        parts.append(f"[Figure Caption] {caption}")
    if ai_desc:
        parts.append(f"[Figure Description] {ai_desc}")
    if plain and len(plain) > 20:
        parts.append(f"[Figure Content] {plain}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Document-level metadata
# ─────────────────────────────────────────────────────────────────────────────

def _extract_doc_metadata(text: str) -> dict:
    meta: dict = {}
    header = text[:3000]

    pdf_link_m = _RE_PDF_LINK.search(header)
    if pdf_link_m:
        meta["source_pdf_name"] = pdf_link_m.group(1).strip()
        meta["source_pdf_url"]  = pdf_link_m.group(2).strip()
    else:
        pdf_plain_m = _RE_PDF_PLAIN.search(header)
        if pdf_plain_m:
            url = pdf_plain_m.group(1).strip()
            meta["source_pdf_url"]  = url
            meta["source_pdf_name"] = Path(url.split("?")[0]).name

    pages_m = _RE_TOTAL_PAGES.search(header)
    if pages_m:
        meta["total_pages"] = int(pages_m.group(1))

    tables_m = _RE_TOTAL_TABLES.search(header)
    if tables_m:
        meta["total_tables"] = int(tables_m.group(1))

    body = text[:8000]
    ver  = _RE_VERSION.search(body)
    if ver:
        meta["doc_version"] = ver.group(1)

    case = _RE_CASE_NUM.search(body)
    if case:
        meta["case_number"] = case.group(1)

    sub = _RE_SUBMITTED_META.search(text)
    if sub:
        meta["submitted"] = sub.group(1).strip()

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Atomic block types used during parsing
# ─────────────────────────────────────────────────────────────────────────────

class _Text:
    """Plain paragraph / list content."""
    __slots__ = ("text",)
    def __init__(self, t: str): self.text = t

class _Table:
    """Already-rendered table text (markdown pipe or HTML → text).
    Marked atomic so it is never split across chunk boundaries."""
    __slots__ = ("content",)
    def __init__(self, c: str): self.content = c

class _Figure:
    """Already-rendered figure text.
    Marked atomic so it is never split across chunk boundaries."""
    __slots__ = ("content",)
    def __init__(self, c: str): self.content = c

_Block = _Text | _Table | _Figure


# ─────────────────────────────────────────────────────────────────────────────
# Section emitter  — the core chunking logic
# ─────────────────────────────────────────────────────────────────────────────
#
# All blocks collected under one heading are packed greedily into
# CHUNK_MAX_TOKENS windows.  Tables and figures are treated as atomic:
# if one doesn't fit in the current window it is moved to the next one.
# Plain text that is itself oversized is sub-split with overlap.
#
# is_table / is_figure are always False in v5+ (tables/figures are inline).
# has_table / has_figure / table_count / figure_count still reflect presence.
# ─────────────────────────────────────────────────────────────────────────────

def _emit_section(
    blocks:              list[_Block],
    stack:               list[str],
    page_start:          Optional[int],
    page_end:            Optional[int],
    source:              str,
    doc_meta:            dict,
    start_index:         int,
    pdf_url:             str = "",
    pdf_name:            str = "",
    energy_utility_name: str = "",       # ← NEW v6
) -> tuple[list[Chunk], int]:
    if not blocks:
        return [], start_index

    idx           = start_index
    level         = len(stack)
    topic         = stack[0] if level >= 1 else ""
    subtopic      = stack[1] if level >= 2 else ""
    section_title = stack[-1] if stack else ""
    section       = " > ".join(stack) if stack else "Document"

    max_chars     = CHUNK_MAX_TOKENS     * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    # ── Step 1: normalise blocks into (text, is_atomic) pairs ─────────────
    # Consecutive _Text blocks are merged eagerly.
    pieces: list[tuple[str, bool]] = []
    pending_text = ""

    def _push_text(t: str) -> None:
        nonlocal pending_text
        if t.strip():
            pending_text = (pending_text + "\n\n" + t).strip() if pending_text else t.strip()

    def _flush_text_piece() -> None:
        nonlocal pending_text
        if pending_text:
            pieces.append((pending_text, False))
            pending_text = ""

    for block in blocks:
        if isinstance(block, _Text):
            _push_text(block.text)
        else:
            _flush_text_piece()
            content = block.content.strip()
            if content:
                pieces.append((content, True))   # atomic

    _flush_text_piece()

    if not pieces:
        return [], idx

    # ── Step 2: greedy packing into windows ───────────────────────────────
    # Each window is a list of (text, is_atomic) tuples.
    windows: list[list[tuple[str, bool]]] = []
    current_window: list[tuple[str, bool]] = []
    current_len = 0

    def _close_window() -> None:
        nonlocal current_window, current_len
        if current_window:
            windows.append(current_window)
        current_window = []
        current_len    = 0

    for text, is_atomic in pieces:
        tlen = len(text)

        if is_atomic:
            # Try to fit in current window; if not, start a new one first
            if current_len + tlen > max_chars and current_window:
                _close_window()
            current_window.append((text, True))
            current_len += tlen
            # After adding, if we're over budget flush immediately
            # (an oversized atomic block occupies its own window)
            if current_len >= max_chars:
                _close_window()

        else:
            # Plain text — may need internal splitting
            if current_len + tlen <= max_chars:
                current_window.append((text, False))
                current_len += tlen
            else:
                # Flush current window, carry overlap tail
                overlap_tail = ""
                if current_window:
                    # Build overlap from tail of current window's plain text
                    flat = "\n\n".join(t for t, _ in current_window)
                    overlap_tail = flat[-overlap_chars:].strip()
                    _close_window()

                # If the piece itself is large, split with overlap
                combined = (overlap_tail + "\n\n" + text).strip() if overlap_tail else text
                if len(combined) > max_chars:
                    sub = _split_text_with_overlap(combined)
                    # All but the last sub-chunk become their own windows
                    for sc in sub[:-1]:
                        windows.append([(sc, False)])
                    # Last sub-chunk starts the next current window
                    if sub:
                        current_window = [(sub[-1], False)]
                        current_len    = len(sub[-1])
                else:
                    current_window = [(combined, False)]
                    current_len    = len(combined)

    _close_window()

    # ── Step 3: render windows → Chunk objects ────────────────────────────
    chunks: list[Chunk] = []

    for window in windows:
        raw = "\n\n".join(t for t, _ in window).strip()
        if not raw:
            continue

        _has_table  = (
            "|" in raw or
            bool(re.search(r'(?m)^[A-Z ]{4,}:\s', raw))
        )
        _has_figure = bool(
            re.search(r'\[Figure Caption\]|\[Figure Description\]|\[Figure Content\]', raw)
        )
        _table_count  = 1 if _has_table  else 0
        _figure_count = 1 if _has_figure else 0

        ctype = _detect_content_type(raw)
        meta  = {
            **doc_meta,
            "char_count":     len(raw),
            "token_estimate": _tokens(raw),
        }

        chunks.append(Chunk(
            chunk_id             = _make_id(source, idx, raw),
            source               = source,
            chunk_index          = idx,
            page_start           = page_start,
            page_end             = page_end,
            heading_level        = level,
            topic                = topic,
            subtopic             = subtopic,
            section_title        = section_title,
            section              = section,
            source_pdf_url       = pdf_url,
            source_pdf_name      = pdf_name,
            region               = FIXED_REGION,        # ← NEW v6
            energy_utility_name  = energy_utility_name, # ← NEW v6
            content              = raw,
            content_type         = ctype,
            has_table            = _has_table,
            has_figure           = _has_figure,
            is_table             = False,   # v5+: tables are always inline
            is_figure            = False,   # v5+: figures are always inline
            table_count          = _table_count,
            figure_count         = _figure_count,
            metadata             = meta,
        ))
        idx += 1

    return chunks, idx


# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_markdown_to_chunks(filepath: str | Path) -> list[Chunk]:  # noqa: C901
    filepath = Path(filepath)
    raw      = filepath.read_text(encoding="utf-8")
    source   = filepath.name
    doc_meta = _extract_doc_metadata(raw)
    pdf_url  = doc_meta.get("source_pdf_url",  "")
    pdf_name = doc_meta.get("source_pdf_name", "")

    # ── NEW v6: derive energy utility name from parent folder ─────────────
    # e.g.  documents/ConEd/file.md  →  "ConEd"
    # If the file sits directly in the root documents folder, parent.name
    # will be the folder name itself (e.g. "documents"); adjust as needed.
    energy_utility_name: str = filepath.parent.name

    chunks:        list[Chunk]           = []
    chunk_index    = 0
    heading_stack: list[str]             = []
    current_page:  Optional[int]         = None
    seen_headers:  set[str]              = set()

    section_blocks: list[_Block]          = []
    section_page_start_val: Optional[int] = None

    in_html_table  = False
    in_figure      = False
    in_pipe_table  = False
    html_table_buf: list[str] = []
    figure_buf:     list[str] = []
    pipe_table_buf: list[str] = []

    def flush_section() -> None:
        nonlocal chunk_index, section_page_start_val
        new_chunks, chunk_index = _emit_section(
            section_blocks,
            heading_stack,
            section_page_start_val,
            current_page,
            source,
            doc_meta,
            chunk_index,
            pdf_url             = pdf_url,
            pdf_name            = pdf_name,
            energy_utility_name = energy_utility_name,  # ← NEW v6
        )
        chunks.extend(new_chunks)
        section_blocks.clear()
        section_page_start_val = current_page

    for raw_line in raw.splitlines(keepends=True):
        stripped = raw_line.strip()

        # ── Fast-path: empty line ──────────────────────────────────────────
        if not stripped:
            if not in_html_table and not in_figure and not in_pipe_table:
                section_blocks.append(_Text(""))
            continue

        # ── PageBreak ─────────────────────────────────────────────────────
        if stripped == "<!-- PageBreak -->":
            continue

        # ── PageNumber ────────────────────────────────────────────────────
        if "PageNumber" in stripped:
            pg_m = _RE_PAGE_NUM.search(stripped)
            if pg_m:
                current_page = int(pg_m.group(1))
                if section_page_start_val is None:
                    section_page_start_val = current_page
            continue

        # ── HTML comments (PageHeader / PageFooter / other) ────────────────
        if stripped.startswith("<!--"):
            if "PageHeader" in stripped:
                hv_m = _RE_PAGE_HDR_VAL.search(stripped)
                if hv_m:
                    hv = hv_m.group(1)
                    if hv in seen_headers:
                        continue
                    seen_headers.add(hv)
                continue
            if "PageFooter" in stripped:
                sub_m = _RE_SUBMITTED.search(stripped)
                if sub_m and "submitted" not in doc_meta:
                    doc_meta["submitted"] = sub_m.group(1).strip()
                continue
            if in_figure:
                figure_buf.append(raw_line)
            continue

        sl = stripped.lower()

        # ══════════════════════════════════════════════════════════════════
        # FIGURE  (multi-line HTML block)
        # ══════════════════════════════════════════════════════════════════
        if sl.startswith("<figure"):
            in_figure  = True
            figure_buf = [raw_line]
            continue

        if in_figure:
            figure_buf.append(raw_line)
            if sl.startswith("</figure"):
                in_figure = False
                fig_text  = _figure_to_text("".join(figure_buf))
                if fig_text.strip():
                    section_blocks.append(_Figure(fig_text))  # inline — never a standalone chunk
                figure_buf = []
            continue

        # ══════════════════════════════════════════════════════════════════
        # HTML TABLE  (multi-line)
        # ══════════════════════════════════════════════════════════════════
        if sl.startswith("<table"):
            in_html_table  = True
            html_table_buf = [raw_line]
            continue

        if in_html_table:
            html_table_buf.append(raw_line)
            if sl.startswith("</table"):
                in_html_table = False
                html  = "".join(html_table_buf)
                rows  = _parse_html_table(html)
                if _is_process_table(rows):
                    text = _process_table_to_text(rows)
                else:
                    text = _generic_table_to_text(rows)
                if text.strip():
                    section_blocks.append(_Table(text))  # inline — never a standalone chunk
                html_table_buf = []
            continue

        # ══════════════════════════════════════════════════════════════════
        # MARKDOWN PIPE TABLE
        # ══════════════════════════════════════════════════════════════════
        if stripped.startswith("|"):
            in_pipe_table = True
            pipe_table_buf.append(raw_line)
            continue

        if in_pipe_table:
            in_pipe_table = False
            table_text = "\n".join(ln.rstrip() for ln in pipe_table_buf).strip()
            if table_text:
                section_blocks.append(_Table(table_text))  # inline
            pipe_table_buf = []
            # fall through — process current line normally

        # ══════════════════════════════════════════════════════════════════
        # HEADING  →  flush current section, start a new one
        # ══════════════════════════════════════════════════════════════════
        if stripped.startswith("#"):
            hdg_m = _RE_HEADING.match(stripped)
            if hdg_m:
                flush_section()
                level   = len(hdg_m.group(1))
                hdg_txt = hdg_m.group(2).strip()
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(hdg_txt)
                section_page_start_val = current_page
                continue

        # ══════════════════════════════════════════════════════════════════
        # HORIZONTAL RULE — visual separator only, skip
        # ══════════════════════════════════════════════════════════════════
        if stripped.startswith("-") and _RE_HRULE.match(stripped):
            continue

        # ══════════════════════════════════════════════════════════════════
        # PLAIN TEXT / LIST ITEM
        # ══════════════════════════════════════════════════════════════════
        section_blocks.append(_Text(stripped))

    # ── Flush remaining open structures ───────────────────────────────────
    if in_pipe_table and pipe_table_buf:
        table_text = "\n".join(ln.rstrip() for ln in pipe_table_buf).strip()
        if table_text:
            section_blocks.append(_Table(table_text))

    flush_section()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_FILE = "manifest.json"


def _cache_dir(folder: Path) -> Path:
    d = folder / ".chunk_cache"
    d.mkdir(exist_ok=True)
    return d


def _load_manifest(cache: Path) -> dict:
    mf = cache / _MANIFEST_FILE
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(cache: Path, manifest: dict) -> None:
    (cache / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _chunk_cache_path(cache: Path, filepath: Path) -> Path:
    stem = _RE_UNSAFE_KEY.sub('_', filepath.stem).strip('_')
    return cache / f"{stem}_chunks.json"


def _file_signature(filepath: Path) -> dict:
    st = filepath.stat()
    return {"mtime": st.st_mtime, "size": st.st_size}


def _is_cached(manifest: dict, filepath: Path) -> bool:
    key = filepath.name
    if key not in manifest:
        return False
    sig     = manifest[key].get("sig", {})
    current = _file_signature(filepath)
    return sig.get("mtime") == current["mtime"] and sig.get("size") == current["size"]


def _save_file_chunks(cache: Path, filepath: Path, chunks: list[Chunk],
                      manifest: dict) -> None:
    cache_path = _chunk_cache_path(cache, filepath)
    cache_path.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    manifest[filepath.name] = {
        "sig":    _file_signature(filepath),
        "chunks": len(chunks),
        "cache":  cache_path.name,
    }


def _load_file_chunks(cache: Path, filepath: Path, manifest: dict) -> list[Chunk]:
    cache_path = cache / manifest[filepath.name]["cache"]
    raw_list   = json.loads(cache_path.read_text(encoding="utf-8"))
    return [Chunk(**d) for d in raw_list]


# ─────────────────────────────────────────────────────────────────────────────
# Parallel batch processor
# ─────────────────────────────────────────────────────────────────────────────

def _parse_file_worker(filepath: Path) -> tuple[str, list[Chunk], int]:
    chunks = parse_markdown_to_chunks(filepath)
    return filepath.name, chunks, len(chunks)


def chunk_directory(
    folder:    str | Path,
    glob:      str  = "**/*.md",
    workers:   int | None = None,
    parallel:  bool = True,
    use_cache: bool = True,
    force:     bool = False,
) -> list[Chunk]:
    """
    Parse all markdown files in *folder* and return a flat list of Chunk objects.

    Checkpoint / resume
    -------------------
    When use_cache=True (default) each file's chunks are saved to
    .chunk_cache/<stem>_chunks.json after parsing.  On the next run, files
    whose mtime + size are unchanged are loaded from cache instead of
    re-parsed.  Pass force=True to ignore the cache and re-chunk everything.

    Folder structure expected (v6)
    ------------------------------
    documents/
      <Energy_Utility_Name>/     ← folder name becomes energy_utility_name
        file1.md
        file2.md
      <Another_Utility>/
        file3.md
    """
    folder = Path(folder)
    files  = sorted(folder.glob(glob))
    print(f"Found {len(files)} markdown file(s) in '{folder}'")

    if not files:
        return []

    cache    = _cache_dir(folder)
    manifest = _load_manifest(cache) if use_cache and not force else {}

    todo:   list[Path] = []
    cached: list[Path] = []

    for f in files:
        if use_cache and not force and _is_cached(manifest, f):
            cached.append(f)
        else:
            todo.append(f)

    if cached:
        print(f"  ✅  {len(cached)} file(s) loaded from cache, "
              f"{len(todo)} file(s) need chunking")
    if not todo:
        all_chunks: list[Chunk] = []
        for f in files:
            fc = _load_file_chunks(cache, f, manifest)
            print(f"  [cache] {f.name}: {len(fc)} chunks")
            all_chunks.extend(fc)
        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks

    fresh_results: dict[str, list[Chunk]] = {}

    if not parallel or len(todo) == 1:
        for f in todo:
            fc = parse_markdown_to_chunks(f)
            print(f"  [new]   {f.name}: {len(fc)} chunks  "
                  f"[utility: {f.parent.name}]")
            if use_cache:
                _save_file_chunks(cache, f, fc, manifest)
                _save_manifest(cache, manifest)
            fresh_results[f.name] = fc
    else:
        import multiprocessing
        max_workers = workers or min(multiprocessing.cpu_count(), len(todo))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_parse_file_worker, f): f for f in todo}
            for future in as_completed(futures):
                name, fc, count = future.result()
                f = futures[future]
                print(f"  [new]   {name}: {count} chunks  "
                      f"[utility: {f.parent.name}]")
                if use_cache:
                    _save_file_chunks(cache, f, fc, manifest)
                    _save_manifest(cache, manifest)
                fresh_results[name] = fc

    all_chunks = []
    for f in files:
        if f.name in fresh_results:
            all_chunks.extend(fresh_results[f.name])
        else:
            fc = _load_file_chunks(cache, f, manifest)
            print(f"  [cache] {f.name}: {len(fc)} chunks")
            all_chunks.extend(fc)

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def clear_cache(folder: str | Path) -> None:
    """Delete all cached chunk files for a folder."""
    import shutil
    cache = Path(folder) / ".chunk_cache"
    if cache.exists():
        shutil.rmtree(cache)
        print(f"🗑️  Cache cleared: {cache}")
    else:
        print(f"ℹ️  No cache found at {cache}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    args       = sys.argv[1:]
    force_flag = "--force"    in args
    no_cache   = "--no-cache" in args
    clear_flag = "--clear"    in args
    args       = [a for a in args if not a.startswith("--")]
    folder     = args[0] if args else "./documents"

    if clear_flag:
        clear_cache(folder)
        sys.exit(0)

    t0     = time.perf_counter()
    chunks = chunk_directory(
        folder,
        use_cache = not no_cache,
        force     = force_flag,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n⏱  {len(chunks)} chunks ready in {elapsed:.2f}s")

    # Quick sanity check — show unique utilities found
    utilities = sorted({c.energy_utility_name for c in chunks if c.energy_utility_name})
    print(f"Utilities found: {utilities}")

    out = Path("chunks_preview.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([c.to_dict() for c in chunks[:50]], fh, indent=2,
                  ensure_ascii=False)
    print(f"First 50 chunks → {out}")