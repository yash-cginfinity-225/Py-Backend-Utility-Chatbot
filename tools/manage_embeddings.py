#!/usr/bin/env python3
"""
manage_embeddings.py
--------------------
Admin tool to download, inspect, edit, and re-upload embed cache files.
Optionally reindex affected chunks in Azure AI Search.

Usage (from Backend/):
  # Download embed cache for inspection/editing
  python -m tools.manage_embeddings download --workdir ./embed_edit

  # After editing, re-upload
  python -m tools.manage_embeddings upload --workdir ./embed_edit

  # Reindex specific chunk_ids after vector edits
  python -m tools.manage_embeddings reindex --workdir ./embed_edit --chunk-ids id1 id2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure Backend/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from blob_handler import (
    download_blob_from_container,
    upload_blob_to_container,
    list_blobs_with_prefix,
    MARKDOWN_CONTAINER,
)

EMBED_CACHE_CONTAINER = os.environ.get("EMBED_CACHE_CONTAINER", None)
EMBED_CACHE_PREFIX = (os.environ.get("EMBED_CACHE_PREFIX", "embed_cache")).rstrip("/") + "/"
CHUNK_CACHE_PREFIX = os.environ.get("CHUNK_CACHE_PREFIX", "chunk_cache/")
CHUNK_CACHE_CONTAINER = os.environ.get("CHUNK_CACHE_CONTAINER", None)

_VECTORS_FILE  = "embeddings.jsonl"
_MANIFEST_FILE = "manifest.json"


def cmd_download(args):
    """Download embed cache files from blob for inspection/editing."""
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for fname in (_VECTORS_FILE, _MANIFEST_FILE):
        blob_path = f"{EMBED_CACHE_PREFIX}{fname}"
        try:
            data = download_blob_from_container(blob_path, container_name=EMBED_CACHE_CONTAINER)
            (workdir / fname).write_bytes(data)
            print(f"✓ Downloaded {fname} → {workdir / fname}")
        except Exception as e:
            print(f"⚠ Could not download {fname}: {e}")

    # Show stats
    vec_path = workdir / _VECTORS_FILE
    if vec_path.exists():
        lines = [l for l in vec_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Vectors: {len(lines)}")
    man_path = workdir / _MANIFEST_FILE
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        uploaded = sum(1 for v in manifest.values() if v.get("uploaded"))
        print(f"  Manifest entries: {len(manifest)} ({uploaded} uploaded)")


def cmd_upload(args):
    """Upload edited embed cache files back to blob."""
    workdir = Path(args.workdir)

    for fname in (_VECTORS_FILE, _MANIFEST_FILE):
        local_path = workdir / fname
        if not local_path.exists():
            print(f"⚠ {fname} not found in {workdir}")
            continue
        blob_path = f"{EMBED_CACHE_PREFIX}{fname}"
        upload_blob_to_container(blob_path, local_path.read_bytes(),
                                 container_name=EMBED_CACHE_CONTAINER)
        print(f"✓ Uploaded {fname} → {blob_path}")


def cmd_reindex(args):
    """Reindex specific chunks after editing their vectors."""
    workdir = Path(args.workdir)
    chunk_ids = args.chunk_ids

    if not chunk_ids:
        print("No chunk_ids specified.")
        return

    # Load edited vectors
    vec_path = workdir / _VECTORS_FILE
    if not vec_path.exists():
        print(f"⚠ {vec_path} not found")
        return

    vectors: dict[str, list[float]] = {}
    for line in vec_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entry = json.loads(line)
            vectors[entry["chunk_id"]] = entry["vector"]

    # Import indexer helpers
    from chunker import Chunk
    from indexer import _get_clients, _chunk_to_doc

    _, search_client, _, _ = _get_clients()

    # Find chunk data from chunk_cache blobs
    target_ids = set(chunk_ids)
    found_chunks: dict[str, Chunk] = {}

    cache_blobs = list_blobs_with_prefix(CHUNK_CACHE_PREFIX, container_name=CHUNK_CACHE_CONTAINER)
    for blob_name in cache_blobs:
        if not blob_name.endswith("_chunks.json"):
            continue
        data = download_blob_from_container(blob_name, container_name=CHUNK_CACHE_CONTAINER)
        chunk_list = json.loads(data.decode("utf-8"))
        for cd in chunk_list:
            if cd["chunk_id"] in target_ids:
                found_chunks[cd["chunk_id"]] = Chunk(**cd)

    # Build documents and reindex
    docs = []
    for cid in chunk_ids:
        if cid not in found_chunks:
            print(f"⚠ Chunk {cid} not found in cache")
            continue
        if cid not in vectors:
            print(f"⚠ No vector for {cid} in {_VECTORS_FILE}")
            continue
        doc = _chunk_to_doc(found_chunks[cid], vectors[cid])
        docs.append(doc)

    if docs:
        results = search_client.upload_documents(documents=docs)
        ok = sum(1 for r in results if r.succeeded)
        fail = sum(1 for r in results if not r.succeeded)
        print(f"✓ Reindexed {ok} chunk(s)" + (f", {fail} failed" if fail else ""))
    else:
        print("Nothing to reindex.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage embed cache in blob storage")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download embed cache from blob")
    dl.add_argument("--workdir", default="./embed_edit")

    ul = sub.add_parser("upload", help="Upload edited embed cache to blob")
    ul.add_argument("--workdir", default="./embed_edit")

    ri = sub.add_parser("reindex", help="Reindex chunks after vector edits")
    ri.add_argument("--workdir", default="./embed_edit")
    ri.add_argument("--chunk-ids", nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "reindex":
        cmd_reindex(args)
    else:
        parser.print_help()
