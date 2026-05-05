#!/usr/bin/env python3
"""
sync_chunk_cache.py
-------------------
Startup sync script: downloads chunk cache files from blob storage into a local
.chunk_cache/ directory so the chunker can reuse them.

Usage (from Backend/):
  python -m tools.sync_chunk_cache --documents ./documents
  python -m tools.sync_chunk_cache --documents ./documents --force
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure Backend/ is on sys.path so blob_handler can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from blob_handler import (
    download_blob_from_container,
    list_blobs_with_prefix,
    MARKDOWN_CONTAINER,
)

CHUNK_CACHE_PREFIX = os.environ.get("CHUNK_CACHE_PREFIX", "chunk_cache/")
CHUNK_CACHE_CONTAINER = os.environ.get("CHUNK_CACHE_CONTAINER", None)


def sync_chunk_cache(documents_folder: str, force: bool = False):
    docs = Path(documents_folder)
    dest = docs / ".chunk_cache"
    dest.mkdir(parents=True, exist_ok=True)

    container = CHUNK_CACHE_CONTAINER

    # 1. Try to download remote manifest
    manifest_blob = f"{CHUNK_CACHE_PREFIX}manifest.json"
    remote_manifest: dict = {}
    try:
        raw = download_blob_from_container(manifest_blob, container_name=container)
        remote_manifest = json.loads(raw.decode("utf-8"))
        print(f"✓ Downloaded remote manifest ({len(remote_manifest)} entries)")
    except Exception:
        print("⚠ Remote manifest not found — falling back to blob listing")
        # Fallback: list all chunk blobs and download them
        blobs = list_blobs_with_prefix(CHUNK_CACHE_PREFIX, container_name=container)
        for blob_name in blobs:
            if blob_name.endswith("_chunks.json"):
                fname = Path(blob_name).name
                local_path = dest / fname
                if not local_path.exists() or force:
                    data = download_blob_from_container(blob_name, container_name=container)
                    local_path.write_bytes(data)
                    print(f"  ↓ {fname}")
        print(f"Done — synced chunk files to {dest}")
        return

    # 2. For each manifest entry, download the cache file
    local_manifest: dict = {}
    downloaded = 0
    skipped = 0

    for md_filename, entry in remote_manifest.items():
        cache_filename = entry.get("cache", "")
        if not cache_filename:
            continue

        local_cache_path = dest / cache_filename
        remote_blob = f"{CHUNK_CACHE_PREFIX}{cache_filename}"

        # Check if local file already exists and matches
        if local_cache_path.exists() and not force:
            skipped += 1
            local_manifest[md_filename] = entry
            continue

        # Download
        try:
            data = download_blob_from_container(remote_blob, container_name=container)
            local_cache_path.write_bytes(data)
            local_manifest[md_filename] = entry
            downloaded += 1
            print(f"  ↓ {cache_filename}")
        except Exception as e:
            print(f"  ⚠ Failed to download {cache_filename}: {e}")

    # 3. Write local manifest (same schema as remote — used by chunker)
    local_manifest_path = dest / "manifest.json"
    local_manifest_path.write_text(
        json.dumps(local_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✓ Sync complete: {downloaded} downloaded, {skipped} skipped (already local)")
    print(f"  Local manifest: {local_manifest_path} ({len(local_manifest)} entries)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sync chunk cache from blob to local")
    p.add_argument("--documents", required=True, help="Path to local documents folder")
    p.add_argument("--force", action="store_true", help="Re-download even if local files exist")
    args = p.parse_args()
    sync_chunk_cache(args.documents, force=args.force)
