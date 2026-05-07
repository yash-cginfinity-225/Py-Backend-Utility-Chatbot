"""
blob_handler.py
---------------
Azure Blob Storage helper for the file-upload pipeline.

Two containers
==============
AZURE_UPLOAD_CONTAINER   – files uploaded by the UI, one directory per client
AZURE_MARKDOWN_CONTAINER – markdown output from the processing pipeline

Each container stores blobs under  <client_name>/… virtual directories.
Root-level blobs (no '/' in the name) are ignored when listing clients.
"""

import os
from typing import BinaryIO

from azure.storage.blob import BlobPrefix, BlobServiceClient, ContainerClient
from dotenv import load_dotenv

load_dotenv()

CONN_STR            = os.environ["AZURE_BLOB_CONNECTION_STRING"]
UPLOAD_CONTAINER    = os.environ["AZURE_UPLOAD_CONTAINER"]
MARKDOWN_CONTAINER  = os.environ["AZURE_MARKDOWN_CONTAINER"]


def _container(name: str) -> ContainerClient:
    service = BlobServiceClient.from_connection_string(CONN_STR)
    return service.get_container_client(name)


def _upload_container() -> ContainerClient:
    return _container(UPLOAD_CONTAINER)


def _markdown_container() -> ContainerClient:
    return _container(MARKDOWN_CONTAINER)


# ── Client helpers ────────────────────────────────────────────────────────────

def list_clients() -> list[str]:
    """Return client names (top-level virtual directories) in the upload
    container.  Root-level blobs (files without a '/') are ignored."""
    cc = _upload_container()
    clients: list[str] = []
    for item in cc.walk_blobs(delimiter="/"):
        if isinstance(item, BlobPrefix):
            # item.name looks like "ClientA/"
            clients.append(item.name.rstrip("/"))
    return sorted(clients)


def create_client(name: str) -> None:
    """Create a placeholder .keep blob for *name* in both containers."""
    for cc in (_upload_container(), _markdown_container()):
        placeholder = f"{name}/.keep"
        blob = cc.get_blob_client(placeholder)
        if not _blob_exists(blob):
            blob.upload_blob(b"", overwrite=True)


def _blob_exists(blob_client) -> bool:
    try:
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False


# ── Upload helpers ────────────────────────────────────────────────────────────

def upload_file(client_name: str, filename: str, data: BinaryIO) -> str:
    """Upload a file into  <upload_container>/<client_name>/<filename>.
    Returns the blob path."""
    cc = _upload_container()
    blob_path = f"{client_name}/{filename}"
    blob = cc.get_blob_client(blob_path)
    blob.upload_blob(data, overwrite=True)
    return blob_path


def upload_markdown(client_name: str, filename: str, data: bytes) -> str:
    """Upload a markdown file into  <markdown_container>/<client_name>/<filename>.
    Returns the blob path."""
    cc = _markdown_container()
    blob_path = f"{client_name}/{filename}"
    blob = cc.get_blob_client(blob_path)
    blob.upload_blob(data, overwrite=True)
    return blob_path


# ── Download / list helpers ───────────────────────────────────────────────────

def list_uploaded_files(client_name: str, include_processed: bool = False) -> list[str]:
    """Return filenames under <upload_container>/<client_name>/.
    By default, skip files that start with 'r-' (already processed)."""
    cc = _upload_container()
    prefix = f"{client_name}/"
    names: list[str] = []
    for blob in cc.list_blobs(name_starts_with=prefix):
        fname = blob.name[len(prefix):]
        if not fname or fname == ".keep" or "/" in fname:
            continue
        if not include_processed and fname.startswith("read-"):
            continue
        names.append(fname)
    return sorted(names)


def list_processed_files(client_name: str) -> list[str]:
    """Return display names of processed (read-*) files under
    <upload_container>/<client_name>/."""
    cc = _upload_container()
    prefix = f"{client_name}/"
    names: list[str] = []
    for blob in cc.list_blobs(name_starts_with=prefix):
        fname = blob.name[len(prefix):]
        if not fname or fname == ".keep" or "/" in fname:
            continue
        if fname.startswith("read-"):
            # Strip the "read-" prefix for display
            names.append(fname[len("read-"):])
    return sorted(names)


def list_all_files(client_name: str) -> list[str]:
    """Return display names of ALL files under <upload_container>/<client_name>/.
    Both unprocessed and processed (read-*) files are included.
    When a file exists in both forms the display name is deduplicated."""
    cc = _upload_container()
    prefix = f"{client_name}/"
    seen: set[str] = set()
    for blob in cc.list_blobs(name_starts_with=prefix):
        fname = blob.name[len(prefix):]
        if not fname or fname == ".keep" or "/" in fname:
            continue
        display = fname[len("read-"):] if fname.startswith("read-") else fname
        seen.add(display)
    return sorted(seen)


def download_blob(blob_path: str) -> bytes:
    """Download a blob from the upload container by its full path."""
    cc = _upload_container()
    blob = cc.get_blob_client(blob_path)
    return blob.download_blob().readall()


# ── Rename (mark processed) ──────────────────────────────────────────────────

def rename_to_processed(client_name: str, filename: str) -> str:
    """Copy  <client>/<filename>  →  <client>/read-<filename>
    in the upload container, then delete the original.
    Returns the new blob path."""
    cc = _upload_container()
    src_path = f"{client_name}/{filename}"
    dst_path = f"{client_name}/read-{filename}"

    src_blob = cc.get_blob_client(src_path)
    dst_blob = cc.get_blob_client(dst_path)

    dst_blob.start_copy_from_url(src_blob.url)
    src_blob.delete_blob()
    return dst_path


# ── Generic blob helpers (any container) ──────────────────────────────────────

def upload_blob_to_container(blob_path: str, data: bytes,
                             container_name: str | None = None) -> str:
    """Upload bytes to any blob path in the given container (default: markdown).
    Overwrites if exists. Returns the blob path."""
    cc = _container(container_name or MARKDOWN_CONTAINER)
    blob = cc.get_blob_client(blob_path)
    blob.upload_blob(data, overwrite=True)
    return blob_path


def download_blob_from_container(blob_path: str,
                                 container_name: str | None = None) -> bytes:
    """Download a blob by path from the given container (default: markdown)."""
    cc = _container(container_name or MARKDOWN_CONTAINER)
    blob = cc.get_blob_client(blob_path)
    return blob.download_blob().readall()


def list_blobs_with_prefix(prefix: str,
                           container_name: str | None = None) -> list[str]:
    """List blob names under a prefix in the given container (default: markdown)."""
    cc = _container(container_name or MARKDOWN_CONTAINER)
    return [b.name for b in cc.list_blobs(name_starts_with=prefix)]


def blob_exists_in_container(blob_path: str,
                             container_name: str | None = None) -> bool:
    """Check whether a blob exists in the given container (default: markdown)."""
    cc = _container(container_name or MARKDOWN_CONTAINER)
    blob = cc.get_blob_client(blob_path)
    return _blob_exists(blob)
