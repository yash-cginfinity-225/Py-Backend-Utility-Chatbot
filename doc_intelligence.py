# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
# from azure.core.credentials import AzureKeyCredential
# import pymupdf
# import io
# import json
# from pathlib import Path
# import os
# from dotenv import load_dotenv
# from figure_description import main as run_figure_description

# load_dotenv()

# AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
# AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

# client = DocumentIntelligenceClient(
#     endpoint=os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
#     credential=AzureKeyCredential(os.getenv("DOCUMENT_INTELLIGENCE_API_KEY"))
# )


# def analyze_pdf(local_path, output_file, blob_relative_path):
#     """
#     Analyze PDF page by page (for free tier) and combine results into one markdown file.
#     Also saves the complete JSON response.
#     """

#     local_path = Path(local_path)
#     output_file = Path(output_file)
#     output_file.parent.mkdir(parents=True, exist_ok=True)

#     # Correct blob URL using folder structure
#     blob_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{blob_relative_path.as_posix()}"

#     # JSON output file
#     json_output_file = output_file.with_suffix('.json')

#     # Open PDF
#     doc = pymupdf.open(local_path)
#     total_pages = len(doc)

#     print(f"\nProcessing {local_path.name} - {total_pages} pages")
#     print("=" * 70)

#     all_content = []
#     all_results = []
#     total_tables = 0
#     total_paragraphs = 0

#     for page_num in range(total_pages):
#         print(f"Processing page {page_num + 1}/{total_pages}...", end=" ")

#         single_page_doc = pymupdf.open()
#         single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

#         pdf_bytes = single_page_doc.write()
#         single_page_doc.close()

#         poller = client.begin_analyze_document(
#             model_id="prebuilt-layout",
#             body=pdf_bytes,
#             output_content_format=DocumentContentFormat.MARKDOWN
#         )

#         result = poller.result()

#         if result.content:
#             all_content.append(f"<!-- Page {page_num + 1} -->\n\n{result.content}")

#         result_dict = result.as_dict()
#         result_dict["page_number"] = page_num + 1
#         all_results.append(result_dict)

#         tables_count = len(result.tables) if result.tables else 0
#         paragraphs_count = len(result.paragraphs) if result.paragraphs else 0

#         total_tables += tables_count
#         total_paragraphs += paragraphs_count

#         print(f"✓ (Tables: {tables_count}, Paragraphs: {paragraphs_count})")

#     doc.close()

#     combined_content = "\n\n---\n\n".join(all_content)

#     with open(output_file, "w", encoding="utf-8") as out:
#         out.write(f"# {local_path.stem}\n\n")
#         out.write(f"**Source PDF:** [{local_path.name}]({blob_url})\n\n")
#         out.write(f"**Total Pages:** {total_pages}  \n")
#         out.write(f"**Total Tables:** {total_tables}  \n")
#         out.write(f"**Total Paragraphs:** {total_paragraphs}  \n\n")
#         out.write("---\n\n")
#         out.write(combined_content)

#     json_data = {
#         "source_file": local_path.name,
#         "total_pages": total_pages,
#         "total_tables": total_tables,
#         "total_paragraphs": total_paragraphs,
#         "pages": all_results
#     }

#     with open(json_output_file, "w", encoding="utf-8") as json_out:
#         json.dump(json_data, json_out, indent=2, ensure_ascii=False)

#     print("=" * 70)
#     print("✓ Complete!")
#     print(f"Markdown saved to: {output_file}")
#     print(f"JSON saved to: {json_output_file}")
#     print(f"Total Tables: {total_tables}")
#     print(f"Total Paragraphs: {total_paragraphs}")

#     print("\n" + "=" * 70)
#     print("Processing figures with AI vision...")
#     print("=" * 70)

#     run_figure_description(
#         pdf_path=str(local_path),
#         json_path=str(json_output_file),
#         output_dir=str(output_file.parent / "figures")
#     )


# # Base directory containing PDFs
# base_dir = Path("C:/Users/NamanMalik/Desktop/US Ocean Passport/utility-chatbot/NY Info")

# for root, dirs, files in os.walk(base_dir):
#     for file in files:
#         if file.endswith(".pdf"):

#             pdf_path = Path(root) / file

#             # Get relative path (preserve folder structure)
#             relative_path = pdf_path.relative_to(base_dir)

#             # Output markdown path
#             output_path = Path("output/document_intelligence") / relative_path.with_suffix(".md")

#             # Skip already processed files
#             if output_path.exists():
#                 print(f"Skipping (already processed): {output_path}")
#                 continue

#             analyze_pdf(pdf_path, output_path, relative_path)



# # analyze_pdf("C:/Users/NamanMalik/Desktop/US Ocean Passport/utility-chatbot/NY Info/Con Edison/Con Edison CUBS.pdf", "output/document_intelligence/Con Edison CUBS.md")

# # for file in os.listdir("C:/Users/NamanMalik/Desktop/US Ocean Passport/utility-chatbot/NY EDI Standards/"):
# #     if file.endswith(".pdf"):
# #         pdf_path = f"C:/Users/NamanMalik/Desktop/US Ocean Passport/utility-chatbot/NY EDI Standards/{file}"
# #         output_path = f"output/document_intelligence/{Path(file).stem}.md"
# #         analyze_pdf(pdf_path, output_path)


import re

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
import json
from pathlib import Path
import os
from dotenv import load_dotenv
try:
    from figure_description import main as run_figure_description  # type: ignore[import-not-found]
except ImportError:
    run_figure_description = None

load_dotenv()

AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_CONTAINER       = os.getenv("AZURE_CONTAINER")

client = DocumentIntelligenceClient(
    endpoint=os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("DOCUMENT_INTELLIGENCE_API_KEY"))
)


def analyze_pdf(local_path, output_file, blob_relative_path):
    """
    Analyze a full PDF in a single Document Intelligence call (paid tier).
    Splits the single flat DI result into per-page entries that match the
    format expected by figure_description.main().

    JSON structure written to disk:
    {
        "source_file": "...",
        "total_pages": N,
        "total_tables": N,
        "total_paragraphs": N,
        "pages": [
            {
                "page_number": 1,          ← 1-indexed
                "content": "...",          ← per-page content slice
                "pages": [...],            ← DI page objects (words/lines/spans)
                "tables": [...],           ← tables on this page
                "paragraphs": [...],       ← paragraphs on this page
            },
            ...
        ]
    }
    """

    local_path  = Path(local_path)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    blob_url        = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{blob_relative_path.as_posix()}"
    json_output_file = output_file.with_suffix('.json')

    print(f"\nProcessing {local_path.name}")
    print("=" * 70)
    print("Sending full PDF to Document Intelligence (single call)...")

    # ── Single full-document DI call ─────────────────────────────────────────
    with open(local_path, "rb") as f:
        pdf_bytes = f.read()

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=pdf_bytes,
        output_content_format=DocumentContentFormat.MARKDOWN
    )

    result     = poller.result()
    result_dict = result.as_dict()

    total_pages      = len(result.pages) if result.pages else 0
    total_tables     = len(result.tables) if result.tables else 0
    total_paragraphs = len(result.paragraphs) if result.paragraphs else 0
    full_content     = result.content or ""

    print(f"✓ DI call complete — {total_pages} pages, {total_tables} tables, {total_paragraphs} paragraphs")

    # ── Split flat content into per-page slices ───────────────────────────────
    # DI returns a single content string with global span offsets for every
    # word/line/paragraph. We split it at page boundaries using the span
    # offsets of the first and last word on each page.
    #
    # Each result.pages[i] has a list of words with span.offset / span.length
    # pointing into the global content string — perfect for slicing.

    di_pages     = result_dict.get("pages", [])
    di_tables    = result_dict.get("tables", [])
    di_paragraphs = result_dict.get("paragraphs", [])

    # Build per-page content slices from global span offsets
    page_content_ranges = []   # list of (start_offset, end_offset) per page

    for i, page in enumerate(di_pages):
        words = page.get("words", [])
        lines = page.get("lines", [])

        offsets = []
        for w in words:
            s = w.get("span", {})
            if "offset" in s and "length" in s:
                offsets.append(s["offset"])
                offsets.append(s["offset"] + s["length"])
        for l in lines:
            for s in l.get("spans", []):
                if "offset" in s and "length" in s:
                    offsets.append(s["offset"])
                    offsets.append(s["offset"] + s["length"])

        if offsets:
            page_start = min(offsets)
            page_end   = max(offsets)
        elif page_content_ranges:
            # No words/lines on this page — use end of previous page + 1
            page_start = page_content_ranges[-1][1] + 1
            page_end   = page_start
        else:
            page_start = 0
            page_end   = 0

        page_content_ranges.append((page_start, page_end))

    # Extend each page's slice to the start of the next page so whitespace
    # and newlines between pages aren't lost
    extended_ranges = []
    for i, (start, end) in enumerate(page_content_ranges):
        if i < len(page_content_ranges) - 1:
            next_start = page_content_ranges[i + 1][0]
            extended_end = next_start  # include everything up to next page start
        else:
            extended_end = len(full_content)
        extended_ranges.append((start, extended_end))

    # ── Build per-page entries ────────────────────────────────────────────────
    def spans_overlap(span_list, page_start, page_end):
        """Return True if any span in span_list overlaps [page_start, page_end]."""
        for s in span_list:
            s_start = s.get("offset", 0)
            s_end   = s_start + s.get("length", 0)
            if s_start < page_end and s_end > page_start:
                return True
        return False

    pages_output = []

    for i, page in enumerate(di_pages):
        page_number          = i + 1
        page_start, page_end = extended_ranges[i]
        page_content         = full_content[page_start:page_end]

        # Filter tables whose bounding regions include this page
        page_tables = [
            t for t in di_tables
            if any(r.get("pageNumber") == page_number
                   for r in t.get("boundingRegions", []))
        ]

        # Filter paragraphs whose spans overlap this page's content range
        page_paragraphs = [
            p for p in di_paragraphs
            if spans_overlap(p.get("spans", []), page_start, page_end)
        ]

        page_entry = {
            "page_number":  page_number,
            "content":      page_content,
            "pages":        [page],          # single DI page object (words/lines)
            "tables":       page_tables,
            "paragraphs":   page_paragraphs,
        }
        pages_output.append(page_entry)

        print(f"  Page {page_number:>3}: {len(page_content):>6} chars | "
              f"{len(page_tables)} tables | {len(page_paragraphs)} paragraphs")

    # ── Write JSON ────────────────────────────────────────────────────────────
    json_data = {
        "source_file":     local_path.name,
        "total_pages":     total_pages,
        "total_tables":    total_tables,
        "total_paragraphs": total_paragraphs,
        "pages":           pages_output,
    }

    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Write the full DI markdown output so downstream steps can read it
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_content)

    print("=" * 70)
    print(f"✓ Markdown saved → {output_file}")
    print(f"✓ JSON saved     → {json_output_file}")
    print(f"  Total pages: {total_pages} | Tables: {total_tables} | Paragraphs: {total_paragraphs}")

    # ── Run figure description pipeline (same as before) ─────────────────────
    if run_figure_description is not None:
        print("\n" + "=" * 70)
        print("Processing figures with AI vision...")
        print("=" * 70)

        run_figure_description(
            pdf_path=str(local_path),
            json_path=str(json_output_file),
            output_dir=str(output_file.parent / "figures"),
            blob_path=blob_relative_path.as_posix()
        )
    else:
        print("\n⚠️  figure_description module not found — skipping figure processing.")

    reprocess_markdown_tables(str(output_file.parent))

from pathlib import Path
import os

def reprocess_markdown_tables(base_dir: str):
    """
    Re-read all markdown files and convert any HTML tables to markdown tables.
    """

    base_dir = Path(base_dir)

    print("\nReprocessing markdown files for HTML table conversion...\n")

    total_files = 0
    updated_files = 0

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):

                md_path = Path(root) / file
                total_files += 1

                with open(md_path, "r", encoding="utf-8") as f:
                    content = f.read()

                new_content = convert_html_tables_to_markdown(content)

                if new_content != content:
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                    updated_files += 1
                    print(f"✓ Updated tables → {md_path}")
                else:
                    print(f"Skipped (no HTML tables) → {md_path}")

    print("\n------------------------------------")
    print(f"Total markdown files scanned: {total_files}")
    print(f"Files updated: {updated_files}")
    print("------------------------------------\n")

def convert_html_tables_to_markdown(content: str) -> str:
    """
    Replace every <table>...</table> block with markdown pipe tables.
    Also removes page markers that accidentally appear inside tables.
    """

    def clean_table_html(html):
        # remove page markers inside tables
        html = re.sub(r'<!--\s*PageNumber="[^"]+"\s*-->', '', html)
        html = re.sub(r'<!--\s*Page\s*\d+\s*-->', '', html)
        html = re.sub(r'\n\s*---\s*\n', '\n', html)
        return html

    def replacer(match):
        table_html = clean_table_html(match.group(0))
        md = html_table_to_markdown(table_html)
        return md if md.strip() else match.group(0)

    return re.sub(
        r'<table[^>]*>.*?</table>',
        replacer,
        content,
        flags=re.DOTALL | re.IGNORECASE
    )

def html_table_to_markdown(html: str) -> str:
    """
    Convert a single HTML <table>...</table> block to a markdown pipe table.

    Handles:
      - <th> header cells  → header row + separator line
      - <td> body cells    → body rows
      - Nested tags stripped from cell content
      - Empty cells preserved
      - Pipe characters in cells escaped
      - Tables with no <th> (row 0 treated as header)
    """
    rows = []
    for row_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE):
        row_html = row_match.group(1)
        has_th   = bool(re.search(r'<th', row_html, re.IGNORECASE))
        cells    = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)

        cleaned = []
        for cell in cells:
            text = re.sub(r'<[^>]+>', ' ', cell)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.replace('|', '\\|')
            cleaned.append(text)

        if cleaned:
            rows.append({"cells": cleaned, "is_header": has_th})

    if not rows:
        return ""

    col_count = max(len(r["cells"]) for r in rows)

    for row in rows:
        while len(row["cells"]) < col_count:
            row["cells"].append("")

    header_idx = next((i for i, r in enumerate(rows) if r["is_header"]), 0)

    col_widths = [3] * col_count
    for row in rows:
        for j, cell in enumerate(row["cells"]):
            col_widths[j] = max(col_widths[j], len(cell))

    def format_row(cells):
        padded = [cells[j].ljust(col_widths[j]) for j in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    def separator_row():
        return "| " + " | ".join("-" * col_widths[j] for j in range(col_count)) + " |"

    lines = []
    for i, row in enumerate(rows):
        lines.append(format_row(row["cells"]))
        if i == header_idx:
            lines.append(separator_row())

    return "\n".join(lines)
# ─── Walk directory and process all PDFs ─────────────────────────────────────

if __name__ == "__main__":
    base_dir = Path("C:/Users/NamanMalik/Desktop/US Ocean Passport/utility-chatbot/NY Info")

    for root, dirs, files in os.walk(base_dir):
        reprocess_markdown_tables("output/document_intelligence")
        for file in files:
            if file.endswith(".pdf"):

                pdf_path      = Path(root) / file
                relative_path = pdf_path.relative_to(base_dir)
                output_path   = Path("output/document_intelligence") / relative_path.with_suffix(".md")

                if output_path.exists():
                    print(f"Skipping (already processed): {output_path}")
                    continue

                analyze_pdf(pdf_path, output_path, relative_path)