"""
process_files.py
----------------
Pipeline that processes uploaded files from Azure Blob Storage.

For each file:
  1.  Download the PDF from blob storage to a local temp directory.
  2.  Run doc_intelligence.analyze_pdf  →  produce Markdown + JSON files.
  3.  Run the chunker                   →  produce Chunk objects.
  4.  Run the indexer pipeline           →  embed + upload chunks to Azure AI Search.
  5.  Upload the Markdown + JSON to     Markdown_Files/<client>/<stem>.md/.json
  6.  Upload chunk cache to             chunk_cache/<client>_<stem>_chunks.json
  7.  Update remote chunk_cache/manifest.json

Can be called programmatically via run_for_files() or from the CLI.

Usage:
  python process_files.py                 # process ALL clients (unread files)
  python process_files.py ClientA         # process only ClientA
  python process_files.py ClientA --recreate   # recreate search index first
"""

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from blob_handler import (
    download_blob,
    download_blob_from_container,
    list_clients,
    list_uploaded_files,
    upload_blob_to_container,
    upload_markdown,
    MARKDOWN_CONTAINER,
)
from doc_intelligence import analyze_pdf
from chunker import Chunk, parse_markdown_to_chunks
from indexer import (
    EmbedCache,
    _get_clients as get_indexer_clients,
    create_index,
    run_pipeline,
    delete_index,
    index_exists,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CHUNK_CACHE_PREFIX = os.environ.get("CHUNK_CACHE_PREFIX", "chunk_cache/")
CHUNK_CACHE_CONTAINER = os.environ.get("CHUNK_CACHE_CONTAINER", None)  # defaults to MARKDOWN_CONTAINER


# ── Chunk cache helpers ───────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    """Make a blob-safe filename component."""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name).strip('_')


def _upload_chunk_cache(client_name: str, md_path: Path, chunks: list[Chunk]) -> str:
    """Upload per-file chunk cache to blob. Returns cache filename."""
    stem = _safe_name(md_path.stem)
    client_safe = _safe_name(client_name)
    cache_filename = f"{client_safe}_{stem}_chunks.json"
    blob_path = f"{CHUNK_CACHE_PREFIX}{cache_filename}"

    data = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False).encode("utf-8")
    upload_blob_to_container(blob_path, data, container_name=CHUNK_CACHE_CONTAINER)
    return cache_filename


def _update_chunk_manifest(md_filename: str, md_path: Path,
                           cache_filename: str, chunk_count: int) -> None:
    """Read remote chunk_cache/manifest.json, update entry, re-upload."""
    manifest_blob = f"{CHUNK_CACHE_PREFIX}manifest.json"
    container = CHUNK_CACHE_CONTAINER

    # Read existing manifest
    manifest: dict = {}
    try:
        raw = download_blob_from_container(manifest_blob, container_name=container)
        manifest = json.loads(raw.decode("utf-8"))
    except Exception:
        pass  # first time or missing — start fresh

    # Build entry matching existing schema
    stat = md_path.stat()
    manifest[md_filename] = {
        "sig": {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        },
        "chunks": chunk_count,
        "cache": cache_filename,
    }

    # Upload updated manifest
    upload_blob_to_container(
        manifest_blob,
        json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        container_name=container,
    )


# ── Main processing loop ─────────────────────────────────────────────────────

def process_client(client_name: str, tmp_root: Path, embed_cache: EmbedCache,
                   search_client, openai_client, embed_deploy,
                   filenames: list[str] | None = None) -> int:
    """Process files for one client.  If *filenames* is given, process exactly
    those blobs; otherwise fall back to all unprocessed files in blob storage.
    Returns count of files processed."""
    files = filenames if filenames is not None else list_uploaded_files(client_name)
    if not files:
        logger.info("Client '%s': no files to process.", client_name)
        return 0

    logger.info("Client '%s': %d file(s) to process.", client_name, len(files))
    processed = 0

    for filename in files:
        blob_path = f"{client_name}/{filename}"
        logger.info("─" * 60)
        logger.info("Processing: %s", blob_path)

        # 1. Download from blob
        pdf_bytes = download_blob(blob_path)

        # Write PDF to a temp file so analyze_pdf can read it from disk
        client_tmp = tmp_root / client_name
        client_tmp.mkdir(parents=True, exist_ok=True)
        local_pdf = client_tmp / filename
        local_pdf.write_bytes(pdf_bytes)

        # 2. Document Intelligence → Markdown + JSON
        md_filename = Path(filename).with_suffix(".md").name
        md_path = client_tmp / md_filename
        blob_relative_path = Path(client_name) / filename
        analyze_pdf(local_pdf, md_path, blob_relative_path)

        # Read the produced markdown
        md_text = md_path.read_text(encoding="utf-8")
        json_path = md_path.with_suffix(".json")

        # 3. Chunk
        chunks = parse_markdown_to_chunks(md_path)
        logger.info("  Chunker produced %d chunks.", len(chunks))

        if chunks:
            # 4. Embed + upload to Azure AI Search
            run_pipeline(chunks, search_client, openai_client, embed_deploy, embed_cache)

        # 5. Upload Markdown + JSON to client/ in markdown container
        upload_markdown(client_name, md_filename, md_text.encode("utf-8"))
        logger.info("  Markdown uploaded → %s/%s", client_name, md_filename)

        if json_path.exists():
            json_blob_name = Path(filename).with_suffix(".json").name
            upload_markdown(client_name, json_blob_name, json_path.read_bytes())
            logger.info("  JSON uploaded     → %s/%s", client_name, json_blob_name)

        # 6. Upload chunk cache (flat) + update manifest
        if chunks:
            cache_filename = _upload_chunk_cache(client_name, md_path, chunks)
            _update_chunk_manifest(md_filename, md_path, cache_filename, len(chunks))
            logger.info("  Chunk cache       → %s%s", CHUNK_CACHE_PREFIX, cache_filename)
            logger.info("  Chunk manifest    → updated")

        processed += 1

    return processed


def main(target_client: str | None = None, recreate: bool = False):
    # Resolve clients to process
    if target_client:
        clients = [target_client]
    else:
        clients = list_clients()
        if not clients:
            logger.info("No clients found in blob storage.")
            return

    # Set up indexer clients
    index_client, search_client, openai_client, embed_deploy = get_indexer_clients()

    if recreate:
        logger.info("♻️  Dropping and recreating search index …")
        delete_index(index_client)
    if not index_exists(index_client):
        create_index(index_client)

    # Shared embed cache — default to /tmp so it works on ephemeral App Service
    # instances. Remote blob is always the authoritative copy; local is just a
    # working directory for the current run.
    cache_dir = Path(os.environ.get("EMBED_CACHE_DIR", "/tmp/embed_cache"))
    embed_cache = EmbedCache(cache_dir)

    total_processed = 0

    with tempfile.TemporaryDirectory(prefix="process_files_") as tmp:
        tmp_root = Path(tmp)
        for client in clients:
            n = process_client(
                client, tmp_root, embed_cache,
                search_client, openai_client, embed_deploy,
            )
            total_processed += n

    logger.info("=" * 60)
    logger.info("Done — %d file(s) processed across %d client(s).",
                total_processed, len(clients))


# ── Programmatic entry point ────────────────────────────────────────────────

def run_for_files(client_name: str, filenames: list[str]) -> None:
    """Process a specific list of already-uploaded files for *client_name*.
    Intended to be called from app.py as a background task immediately after
    files are uploaded to blob storage."""
    if not filenames:
        return

    logger.info("run_for_files: client=%s files=%s", client_name, filenames)

    index_client, search_client, openai_client, embed_deploy = get_indexer_clients()
    if not index_exists(index_client):
        create_index(index_client)

    cache_dir = Path(os.environ.get("EMBED_CACHE_DIR", "/tmp/embed_cache"))
    embed_cache = EmbedCache(cache_dir)

    with tempfile.TemporaryDirectory(prefix="process_files_") as tmp:
        tmp_root = Path(tmp)
        n = process_client(
            client_name, tmp_root, embed_cache,
            search_client, openai_client, embed_deploy,
            filenames=filenames,
        )

    logger.info("run_for_files: done — %d file(s) processed.", n)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    recreate_flag = "--recreate" in args
    args = [a for a in args if not a.startswith("--")]
    client_arg = args[0] if args else None
    main(target_client=client_arg, recreate=recreate_flag)
