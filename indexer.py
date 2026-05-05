# """
# indexer.py  — Azure AI Search Indexer v4
# -----------------------------------------
# Creates / updates the Azure AI Search index schema, then embeds and uploads
# all chunks produced by chunker.py v5.

# NEW in v4: Embedding + upload checkpoint / resume
# ==================================================
# Embeddings are expensive and slow.  If a run is interrupted (crash, rate-limit
# death, Ctrl-C), v4 picks up exactly where it left off:

#   .embed_cache/
#     manifest.json          ← { chunk_id: { "embedded": bool, "uploaded": bool } }
#     embeddings.jsonl       ← one JSON line per embedded chunk:
#                               { "chunk_id": "...", "vector": [...] }

# On restart:
#   • Chunks whose chunk_id already has a vector in embeddings.jsonl are NOT
#     re-embedded (their cached vector is reused).
#   • Chunks already marked uploaded=true in manifest.json are NOT re-uploaded.
#   • Only the remaining chunks are embedded + uploaded.

# Cache lives next to the documents folder by default, or set EMBED_CACHE_DIR.

# Environment variables required
# ================================
#   AZURE_SEARCH_ENDPOINT       e.g. https://my-search.search.windows.net
#   AZURE_SEARCH_ADMIN_KEY
#   AZURE_OPENAI_ENDPOINT       e.g. https://my-aoai.openai.azure.com
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_EMBED_DEPLOY   deployment name, e.g. text-embedding-3-small

# Optional
# ========
#   SEARCH_INDEX_NAME           default: "edi-documents"
#   VECTOR_DIMENSIONS           default: 1536
#   EMBED_CACHE_DIR             default: ".embed_cache" next to documents folder
# """

# import json
# import os
# import re
# import time
# from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv()

# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient
# from azure.search.documents.indexes import SearchIndexClient
# from azure.search.documents.indexes.models import (
#     HnswAlgorithmConfiguration,
#     SearchField,
#     SearchFieldDataType,
#     SearchIndex,
#     SearchableField,
#     SemanticConfiguration,
#     SemanticField,
#     SemanticPrioritizedFields,
#     SemanticSearch,
#     SimpleField,
#     VectorSearch,
#     VectorSearchProfile,
# )
# from openai import AzureOpenAI

# from chunker import Chunk, chunk_directory

# # ─────────────────────────────────────────────────────────────────────────────
# # Configuration
# # ─────────────────────────────────────────────────────────────────────────────

# INDEX_NAME        = os.environ.get("SEARCH_INDEX_NAME", "edi-documents")
# VECTOR_DIMENSIONS = int(os.environ.get("VECTOR_DIMENSIONS", "1536"))
# UPLOAD_BATCH_SIZE = 50    # documents per upload call
# EMBED_BATCH_SIZE  = 4     # texts per embedding API call (S0 tier safe)
# # EMBED_SLEEP_SEC   = 2.0   # pause between embedding batches
# EMBED_MAX_RETRIES = 6

# _EMBED_CACHE_MANIFEST = "manifest.json"
# _EMBED_CACHE_VECTORS  = "embeddings.jsonl"


# # ─────────────────────────────────────────────────────────────────────────────
# # Client factory
# # ─────────────────────────────────────────────────────────────────────────────

# def _get_clients():
#     search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
#     search_key      = os.environ["AZURE_SEARCH_ADMIN_KEY"]
#     aoai_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"]
#     aoai_key        = os.environ["AZURE_OPENAI_API_KEY"]
#     embed_deploy    = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY",
#                                      "text-embedding-3-small")

#     cred          = AzureKeyCredential(search_key)
#     index_client  = SearchIndexClient(search_endpoint, cred)
#     search_client = SearchClient(search_endpoint, INDEX_NAME, cred)
#     openai_client = AzureOpenAI(
#         azure_endpoint=aoai_endpoint,
#         api_key=aoai_key,
#         api_version="2024-02-01",
#     )
#     return index_client, search_client, openai_client, embed_deploy


# # ─────────────────────────────────────────────────────────────────────────────
# # Index schema
# # ─────────────────────────────────────────────────────────────────────────────

# def create_index(index_client: SearchIndexClient) -> None:
#     """Create or update the Azure AI Search index."""

#     fields = [
#         # ── Identity ──────────────────────────────────────────────────────
#         SimpleField(name="chunk_id",        type=SearchFieldDataType.String,
#                     key=True, filterable=True),
#         SimpleField(name="source",          type=SearchFieldDataType.String,
#                     filterable=True, facetable=True),
#         SimpleField(name="source_pdf_url",  type=SearchFieldDataType.String,
#                     filterable=True),
#         SimpleField(name="source_pdf_name", type=SearchFieldDataType.String,
#                     filterable=True, facetable=True),
#         SimpleField(name="chunk_index",     type=SearchFieldDataType.Int32,
#                     filterable=True, sortable=True),

#         # ── Location ──────────────────────────────────────────────────────
#         SimpleField(name="page_start", type=SearchFieldDataType.Int32,
#                     filterable=True, sortable=True),
#         SimpleField(name="page_end",   type=SearchFieldDataType.Int32,
#                     filterable=True, sortable=True),

#         # ── Hierarchy ─────────────────────────────────────────────────────
#         SimpleField(name="heading_level", type=SearchFieldDataType.Int32,
#                     filterable=True, sortable=True),
#         SearchableField(name="topic",         type=SearchFieldDataType.String,
#                         filterable=True, facetable=True,
#                         analyzer_name="en.microsoft"),
#         SearchableField(name="subtopic",      type=SearchFieldDataType.String,
#                         filterable=True, facetable=True,
#                         analyzer_name="en.microsoft"),
#         SearchableField(name="section_title", type=SearchFieldDataType.String,
#                         filterable=True, analyzer_name="en.microsoft"),
#         SearchableField(name="section",       type=SearchFieldDataType.String,
#                         filterable=True, analyzer_name="en.microsoft"),

#         # ── Content ───────────────────────────────────────────────────────
#         SearchableField(name="content", type=SearchFieldDataType.String,
#                         analyzer_name="en.microsoft"),

#         # ── Content-type flags ────────────────────────────────────────────
#         SimpleField(name="content_type",  type=SearchFieldDataType.String,
#                     filterable=True, facetable=True),
#         SimpleField(name="has_table",     type=SearchFieldDataType.Boolean,
#                     filterable=True),
#         SimpleField(name="has_figure",    type=SearchFieldDataType.Boolean,
#                     filterable=True),
#         SimpleField(name="is_table",      type=SearchFieldDataType.Boolean,
#                     filterable=True),
#         SimpleField(name="is_figure",     type=SearchFieldDataType.Boolean,
#                     filterable=True),
#         SimpleField(name="table_count",   type=SearchFieldDataType.Int32,
#                     filterable=True),
#         SimpleField(name="figure_count",  type=SearchFieldDataType.Int32,
#                     filterable=True),

#         # ── Extra metadata as JSON string ─────────────────────────────────
#         SimpleField(name="metadata_json", type=SearchFieldDataType.String),

#         # ── Vector field ──────────────────────────────────────────────────
#         SearchField(
#             name="content_vector",
#             type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
#             searchable=True,
#             vector_search_dimensions=VECTOR_DIMENSIONS,
#             vector_search_profile_name="hnsw-profile",
#         ),
#     ]

#     vector_search = VectorSearch(
#         algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
#         profiles=[VectorSearchProfile(
#             name="hnsw-profile",
#             algorithm_configuration_name="hnsw-config",
#         )],
#     )

#     semantic_search = SemanticSearch(
#         configurations=[SemanticConfiguration(
#             name="semantic-config",
#             prioritized_fields=SemanticPrioritizedFields(
#                 title_field=SemanticField(field_name="section_title"),
#                 keywords_fields=[
#                     SemanticField(field_name="topic"),
#                     SemanticField(field_name="subtopic"),
#                     SemanticField(field_name="section"),
#                 ],
#                 content_fields=[SemanticField(field_name="content")],
#             ),
#         )]
#     )

#     index = SearchIndex(
#         name=INDEX_NAME,
#         fields=fields,
#         vector_search=vector_search,
#         semantic_search=semantic_search,
#     )

#     result = index_client.create_or_update_index(index)
#     print(f"✅  Index '{result.name}' created / updated.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Embedding cache  (new in v4)
# # ─────────────────────────────────────────────────────────────────────────────

# class EmbedCache:
#     """
#     Persistent cache for embeddings and upload status.

#     Layout
#     ------
#     <cache_dir>/
#       manifest.json    — { chunk_id: { "uploaded": bool } }
#       embeddings.jsonl — one line per chunk: { "chunk_id": "...", "vector": [...] }

#     The JSONL file is append-only during a run.  On load, only the LAST entry
#     per chunk_id is kept (so safe to have duplicate lines from crashes).
#     """

#     def __init__(self, cache_dir: Path) -> None:
#         self.cache_dir = cache_dir
#         self.cache_dir.mkdir(parents=True, exist_ok=True)

#         self._manifest_path  = cache_dir / _EMBED_CACHE_MANIFEST
#         self._vectors_path   = cache_dir / _EMBED_CACHE_VECTORS

#         # chunk_id → float list
#         self._vectors:  dict[str, list[float]] = {}
#         # chunk_id → { "uploaded": bool }
#         self._manifest: dict[str, dict]        = {}

#         self._load()

#     # ── Persistence ───────────────────────────────────────────────────────

#     def _load(self) -> None:
#         # Load manifest
#         if self._manifest_path.exists():
#             try:
#                 self._manifest = json.loads(
#                     self._manifest_path.read_text(encoding="utf-8")
#                 )
#             except Exception:
#                 self._manifest = {}

#         # Load vectors (JSONL — last entry per chunk_id wins)
#         if self._vectors_path.exists():
#             try:
#                 for line in self._vectors_path.read_text(encoding="utf-8").splitlines():
#                     line = line.strip()
#                     if not line:
#                         continue
#                     entry = json.loads(line)
#                     self._vectors[entry["chunk_id"]] = entry["vector"]
#             except Exception:
#                 pass  # partial file from crash — whatever was loaded is usable

#         cached_v = len(self._vectors)
#         cached_u = sum(1 for v in self._manifest.values() if v.get("uploaded"))
#         if cached_v or cached_u:
#             print(f"  📦  Embed cache loaded: "
#                   f"{cached_v} vectors, {cached_u} already uploaded")

#     def _save_manifest(self) -> None:
#         self._manifest_path.write_text(
#             json.dumps(self._manifest, ensure_ascii=False),
#             encoding="utf-8",
#         )

#     def _append_vector(self, chunk_id: str, vector: list[float]) -> None:
#         """Append a single embedding to the JSONL file immediately."""
#         with self._vectors_path.open("a", encoding="utf-8") as fh:
#             fh.write(json.dumps({"chunk_id": chunk_id, "vector": vector},
#                                 ensure_ascii=False) + "\n")

#     # ── Public API ────────────────────────────────────────────────────────

#     def has_vector(self, chunk_id: str) -> bool:
#         return chunk_id in self._vectors

#     def get_vector(self, chunk_id: str) -> list[float]:
#         return self._vectors[chunk_id]

#     def save_vector(self, chunk_id: str, vector: list[float]) -> None:
#         """Persist an embedding immediately (crash-safe)."""
#         self._vectors[chunk_id] = vector
#         self._append_vector(chunk_id, vector)

#     def is_uploaded(self, chunk_id: str) -> bool:
#         return self._manifest.get(chunk_id, {}).get("uploaded", False)

#     def mark_uploaded(self, chunk_ids: list[str]) -> None:
#         """Mark a batch of chunk_ids as successfully uploaded, then flush manifest."""
#         for cid in chunk_ids:
#             self._manifest.setdefault(cid, {})["uploaded"] = True
#         self._save_manifest()

#     # ── Stats ─────────────────────────────────────────────────────────────

#     def stats(self) -> tuple[int, int]:
#         """Return (vectors_cached, chunks_uploaded)."""
#         return (
#             len(self._vectors),
#             sum(1 for v in self._manifest.values() if v.get("uploaded")),
#         )

#     def clear(self) -> None:
#         """Delete all cache files."""
#         import shutil
#         shutil.rmtree(self.cache_dir, ignore_errors=True)
#         self._vectors  = {}
#         self._manifest = {}
#         print(f"🗑️  Embed cache cleared: {self.cache_dir}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Embedding  (with per-item cache check)
# # ─────────────────────────────────────────────────────────────────────────────

# def embed_texts_cached(
#     chunks:     list[Chunk],
#     client:     AzureOpenAI,
#     deployment: str,
#     cache:      EmbedCache,
# ) -> dict[str, list[float]]:
#     """
#     Embed all chunks, skipping any whose vector is already in cache.

#     Returns a mapping  chunk_id → vector  for ALL chunks (cached + fresh).
#     """
#     to_embed = [c for c in chunks if not cache.has_vector(c.chunk_id)]
#     cached_n = len(chunks) - len(to_embed)

#     if cached_n:
#         print(f"  ⚡  {cached_n} embeddings loaded from cache, "
#               f"{len(to_embed)} need embedding")

#     # Embed in batches
#     for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
#         batch  = to_embed[i : i + EMBED_BATCH_SIZE]
#         texts  = [c.content for c in batch]

#         for attempt in range(EMBED_MAX_RETRIES):
#             try:
#                 response = client.embeddings.create(
#                     input=texts, model=deployment
#                 )
#                 vectors = [item.embedding for item in response.data]

#                 # Persist each vector immediately — crash-safe
#                 for chunk, vector in zip(batch, vectors):
#                     cache.save_vector(chunk.chunk_id, vector)

#                 done = min(i + EMBED_BATCH_SIZE, len(to_embed))
#                 print(f"    Embedded {done}/{len(to_embed)} …", end="\r")
#                 # time.sleep(EMBED_SLEEP_SEC)
#                 break

#             except Exception as exc:
#                 err = str(exc)
#                 if "429" in err or "RateLimitReached" in err:
#                     wait_m = re.search(
#                         r'retry after (\d+) second', err, re.IGNORECASE
#                     )
#                     wait = int(wait_m.group(1)) if wait_m else (15 * 2 ** attempt)
#                     print(f"\n    ⚠️  Rate-limited. Waiting {wait}s "
#                           f"(attempt {attempt + 1}/{EMBED_MAX_RETRIES}) …")
#                     # time.sleep(wait)
#                 else:
#                     raise
#         else:
#             raise RuntimeError(
#                 f"Embedding failed after {EMBED_MAX_RETRIES} retries "
#                 f"at batch index {i}"
#             )

#     if to_embed:
#         print()  # newline after \r progress

#     # Return full mapping for all chunks
#     return {c.chunk_id: cache.get_vector(c.chunk_id) for c in chunks}


# # ─────────────────────────────────────────────────────────────────────────────
# # Upload  (with per-batch cache check)
# # ─────────────────────────────────────────────────────────────────────────────

# def _chunk_to_doc(chunk: Chunk, vector: list[float]) -> dict:
#     return {
#         "chunk_id":        chunk.chunk_id,
#         "source":          chunk.source,
#         "source_pdf_url":  chunk.source_pdf_url,
#         "source_pdf_name": chunk.source_pdf_name,
#         "chunk_index":     chunk.chunk_index,
#         "page_start":      chunk.page_start or 0,
#         "page_end":        chunk.page_end   or 0,
#         "heading_level":   chunk.heading_level,
#         "topic":           chunk.topic,
#         "subtopic":        chunk.subtopic,
#         "section_title":   chunk.section_title,
#         "section":         chunk.section,
#         "content":         chunk.content,
#         "content_type":    chunk.content_type,
#         "has_table":       chunk.has_table,
#         "has_figure":      chunk.has_figure,
#         "is_table":        chunk.is_table,
#         "is_figure":       chunk.is_figure,
#         "table_count":     chunk.table_count,
#         "figure_count":    chunk.figure_count,
#         "metadata_json":   json.dumps(chunk.metadata, ensure_ascii=False),
#         "content_vector":  vector,
#     }


# def upload_chunks_cached(
#     chunks:        list[Chunk],
#     search_client: SearchClient,
#     openai_client: AzureOpenAI,
#     embed_deploy:  str,
#     cache:         EmbedCache,
# ) -> None:
#     total = len(chunks)

#     # ── Step 1: embed everything (with cache) ─────────────────────────────
#     print(f"\nEmbedding {total} chunks …")
#     vectors = embed_texts_cached(chunks, openai_client, embed_deploy, cache)

#     # ── Step 2: upload in batches, skipping already-uploaded chunks ────────
#     # Filter to chunks not yet uploaded
#     pending = [c for c in chunks if not cache.is_uploaded(c.chunk_id)]
#     skipped = total - len(pending)

#     if skipped:
#         print(f"  ⚡  {skipped} chunks already uploaded, {len(pending)} remaining")

#     if not pending:
#         print("✅  Nothing to upload — all chunks already indexed.")
#         return

#     print(f"Uploading {len(pending)} chunks …")
#     batch_num = 0

#     for batch_start in range(0, len(pending), UPLOAD_BATCH_SIZE):
#         batch     = pending[batch_start : batch_start + UPLOAD_BATCH_SIZE]
#         batch_num += 1

#         docs    = [_chunk_to_doc(c, vectors[c.chunk_id]) for c in batch]
#         results = search_client.upload_documents(documents=docs)

#         succeeded = [c for c, r in zip(batch, results) if r.succeeded]
#         failed    = [c for c, r in zip(batch, results) if not r.succeeded]

#         # Mark succeeded chunks as uploaded immediately
#         if succeeded:
#             cache.mark_uploaded([c.chunk_id for c in succeeded])

#         print(f"  Batch {batch_num}: {len(succeeded)}/{len(batch)} succeeded"
#               + (f"  ⚠️  {len(failed)} failed" if failed else ""))

#     v_total, u_total = cache.stats()
#     print(f"\n✅  Done — {u_total}/{total} chunks indexed into '{INDEX_NAME}'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Index management helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def delete_index(index_client: SearchIndexClient) -> None:
#     try:
#         index_client.delete_index(INDEX_NAME)
#         print(f"🗑️   Index '{INDEX_NAME}' deleted.")
#     except Exception as exc:
#         if "ResourceNotFound" in str(exc) or "404" in str(exc):
#             print(f"ℹ️   Index '{INDEX_NAME}' did not exist — nothing to delete.")
#         else:
#             raise


# def index_exists(index_client: SearchIndexClient) -> bool:
#     try:
#         index_client.get_index(INDEX_NAME)
#         return True
#     except Exception:
#         return False


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# def run_indexer(
#     documents_folder: str,
#     recreate:         bool = False,
#     clear_embed_cache: bool = False,
# ) -> None:
#     index_client, search_client, openai_client, embed_deploy = _get_clients()

#     # Resolve embed cache directory
#     cache_dir_env = os.environ.get("EMBED_CACHE_DIR", "")
#     if cache_dir_env:
#         cache_dir = Path(cache_dir_env)
#     else:
#         cache_dir = Path(documents_folder) / ".embed_cache"

#     embed_cache = EmbedCache(cache_dir)

#     if clear_embed_cache:
#         embed_cache.clear()
#         print("♻️   Embed cache cleared.")

#     # Optionally wipe the index
#     if recreate:
#         print("♻️   --recreate flag set: dropping existing index …")
#         delete_index(index_client)
#     elif index_exists(index_client):
#         print(f"ℹ️   Index '{INDEX_NAME}' already exists. "
#               "Pass --recreate to drop and rebuild it.")

#     # Create / refresh schema
#     create_index(index_client)

#     # Chunk all markdown files (chunker has its own cache)
#     chunks = chunk_directory(documents_folder)
#     if not chunks:
#         print("⚠️  No chunks produced — check your documents folder.")
#         return

#     tables  = sum(1 for c in chunks if c.has_table)
#     figures = sum(1 for c in chunks if c.has_figure)
#     print(f"   {len(chunks)} total chunks  |  "
#           f"{tables} with tables  |  {figures} with figures")

#     # Embed + upload (both with resume support)
#     upload_chunks_cached(
#         chunks, search_client, openai_client, embed_deploy, embed_cache
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys

#     # Usage:
#     #   python indexer.py                              → index ./documents
#     #   python indexer.py ./my_docs                    → index ./my_docs
#     #   python indexer.py ./my_docs --recreate         → drop index first
#     #   python indexer.py ./my_docs --clear-embed      → wipe embed cache, re-embed all
#     #   python indexer.py ./my_docs --recreate --clear-embed  → full reset
#     args = sys.argv[1:]

#     recreate_flag    = "--recreate"    in args
#     clear_embed_flag = "--clear-embed" in args
#     args = [a for a in args if not a.startswith("--")]
#     folder = args[0] if args else "./documents"

#     run_indexer(
#         folder,
#         recreate          = recreate_flag,
#         clear_embed_cache = clear_embed_flag,
#     )

"""
indexer.py  — Azure AI Search Indexer v6
-----------------------------------------
Creates / updates the Azure AI Search index schema, then embeds and uploads
all chunks produced by chunker.py v6.

NEW in v6
=========
- Index schema includes two new fields:
    • region              (String, filterable, facetable) — fixed "New York"
    • energy_utility_name (String, filterable, facetable) — from parent folder name
- _chunk_to_doc() maps both new Chunk fields into the upload document.

Environment variables required
================================
  AZURE_SEARCH_ENDPOINT       e.g. https://my-search.search.windows.net
  AZURE_SEARCH_ADMIN_KEY
  AZURE_OPENAI_ENDPOINT       e.g. https://my-aoai.openai.azure.com
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_EMBED_DEPLOY   deployment name, e.g. text-embedding-3-small

Optional
========
  SEARCH_INDEX_NAME           default: "edi-documents"
  VECTOR_DIMENSIONS           default: 1536
  EMBED_CACHE_DIR             default: ".embed_cache" next to documents folder
  PIPELINE_BATCH_SIZE         default: 200
  UPLOAD_BATCH_SIZE           default: 50
  EMBED_BATCH_SIZE            default: 4
  EMBED_SLEEP_SEC             default: 1.0
"""

import json, logging, os, re, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration, SearchField, SearchFieldDataType, SearchIndex,
    SearchableField, SemanticConfiguration, SemanticField, SemanticPrioritizedFields,
    SemanticSearch, SimpleField, VectorSearch, VectorSearchProfile,
)
from openai import AzureOpenAI
from chunker import Chunk, chunk_directory

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

INDEX_NAME          = os.environ.get("SEARCH_INDEX_NAME",   "edi-documents")
VECTOR_DIMENSIONS   = int(os.environ.get("VECTOR_DIMENSIONS",   "1536"))
PIPELINE_BATCH_SIZE = int(os.environ.get("PIPELINE_BATCH_SIZE", "200"))
UPLOAD_BATCH_SIZE   = int(os.environ.get("UPLOAD_BATCH_SIZE",   "50"))
EMBED_BATCH_SIZE    = int(os.environ.get("EMBED_BATCH_SIZE",    "4"))
# EMBED_SLEEP_SEC     = float(os.environ.get("EMBED_SLEEP_SEC",   "1.0"))
EMBED_MAX_RETRIES   = 6
_EMBED_CACHE_MANIFEST = "manifest.json"
_EMBED_CACHE_VECTORS  = "embeddings.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_clients():
    cred = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])
    index_client  = SearchIndexClient(os.environ["AZURE_SEARCH_ENDPOINT"], cred)
    search_client = SearchClient(os.environ["AZURE_SEARCH_ENDPOINT"], INDEX_NAME, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-01",
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    return index_client, search_client, openai_client, embed_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Index schema
# ─────────────────────────────────────────────────────────────────────────────

def create_index(index_client: SearchIndexClient) -> None:
    """Create or update the Azure AI Search index."""

    fields = [
        # ── Identity ──────────────────────────────────────────────────────
        SimpleField(name="chunk_id",        type=SearchFieldDataType.String,
                    key=True, filterable=True),
        SimpleField(name="source",          type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="source_pdf_url",  type=SearchFieldDataType.String,
                    filterable=True),
        SimpleField(name="source_pdf_name", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="chunk_index",     type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),

        # ── Location ──────────────────────────────────────────────────────
        SimpleField(name="page_start", type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),
        SimpleField(name="page_end",   type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),

        # ── Hierarchy ─────────────────────────────────────────────────────
        SimpleField(name="heading_level", type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),
        SearchableField(name="topic",         type=SearchFieldDataType.String,
                        filterable=True, facetable=True,
                        analyzer_name="en.microsoft"),
        SearchableField(name="subtopic",      type=SearchFieldDataType.String,
                        filterable=True, facetable=True,
                        analyzer_name="en.microsoft"),
        SearchableField(name="section_title", type=SearchFieldDataType.String,
                        filterable=True, analyzer_name="en.microsoft"),
        SearchableField(name="section",       type=SearchFieldDataType.String,
                        filterable=True, analyzer_name="en.microsoft"),

        # ── NEW v6: Utility / Geography ───────────────────────────────────
        SimpleField(name="region",               type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="energy_utility_name",  type=SearchFieldDataType.String,
                    filterable=True, facetable=True),

        # ── Content ───────────────────────────────────────────────────────
        SearchableField(name="content", type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),

        # ── Content-type flags ────────────────────────────────────────────
        SimpleField(name="content_type",  type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="has_table",     type=SearchFieldDataType.Boolean,
                    filterable=True),
        SimpleField(name="has_figure",    type=SearchFieldDataType.Boolean,
                    filterable=True),
        SimpleField(name="is_table",      type=SearchFieldDataType.Boolean,
                    filterable=True),
        SimpleField(name="is_figure",     type=SearchFieldDataType.Boolean,
                    filterable=True),
        SimpleField(name="table_count",   type=SearchFieldDataType.Int32,
                    filterable=True),
        SimpleField(name="figure_count",  type=SearchFieldDataType.Int32,
                    filterable=True),

        # ── Extra metadata as JSON string ─────────────────────────────────
        SimpleField(name="metadata_json", type=SearchFieldDataType.String),

        # ── Vector field ──────────────────────────────────────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    result = index_client.create_or_update_index(SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config",
            )],
        ),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="section_title"),
                keywords_fields=[
                    SemanticField(field_name="topic"),
                    SemanticField(field_name="subtopic"),
                    SemanticField(field_name="section"),
                ],
                content_fields=[SemanticField(field_name="content")],
            ),
        )]),
    ))
    print(f"✅  Index '{result.name}' created / updated.")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache
# ─────────────────────────────────────────────────────────────────────────────

class EmbedCache:
    """Embedding cache with remote blob sync.

    On init: downloads remote embeddings.jsonl and manifest.json if local
    files are absent. After writes: uploads to remote blob to keep it current.
    """

    def __init__(self, cache_dir):
        self.cache_dir      = Path(cache_dir)
        self._manifest_path = self.cache_dir / _EMBED_CACHE_MANIFEST
        self._vectors_path  = self.cache_dir / _EMBED_CACHE_VECTORS
        self._vectors  = {}
        self._manifest = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Remote sync config
        self._remote_container = os.environ.get("EMBED_CACHE_CONTAINER", None)
        self._remote_prefix = (os.environ.get("EMBED_CACHE_PREFIX", "embed_cache")).rstrip("/") + "/"

        self._sync_from_remote()
        self._load()

    def _sync_from_remote(self):
        """Download remote cache files into local dir (remote is authoritative).

        Always overwrites local copies so that a fresh App Service instance or
        a scaled-out instance gets the latest state from blob rather than
        whatever stale file might exist on its local disk.
        """
        from blob_handler import download_blob_from_container
        for local_path, remote_filename in [
            (self._vectors_path,  _EMBED_CACHE_VECTORS),
            (self._manifest_path, _EMBED_CACHE_MANIFEST),
        ]:
            try:
                data = download_blob_from_container(
                    f"{self._remote_prefix}{remote_filename}",
                    container_name=self._remote_container,
                )
                local_path.write_bytes(data)
                print(f"  ↓  Synced remote {remote_filename} → local")
            except Exception as exc:
                logger.debug("Embed cache remote sync skipped for %s: %s", remote_filename, exc)
                # blob doesn't exist yet — start fresh on first ever run

    def _upload_to_remote(self, local_path: Path, remote_filename: str):
        """Upload a local cache file to remote blob (best-effort)."""
        from blob_handler import upload_blob_to_container
        try:
            upload_blob_to_container(
                f"{self._remote_prefix}{remote_filename}",
                local_path.read_bytes(),
                container_name=self._remote_container,
            )
            logger.debug("Embed cache remote upload OK: %s", remote_filename)
        except Exception as exc:
            logger.warning("Embed cache remote upload failed for %s: %s", remote_filename, exc)

    def _load(self):
        if self._manifest_path.exists():
            try:
                self._manifest = json.loads(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._manifest = {}
        if self._vectors_path.exists():
            try:
                for line in self._vectors_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        e = json.loads(line)
                        self._vectors[e["chunk_id"]] = e["vector"]
            except Exception:
                pass
        v = len(self._vectors)
        u = sum(1 for e in self._manifest.values() if e.get("uploaded"))
        if v or u:
            print(f"  📦  Embed cache: {v} vectors, {u} already uploaded")

    def _save_manifest(self):
        self._manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        self._upload_to_remote(self._manifest_path, _EMBED_CACHE_MANIFEST)

    def _append_vector(self, chunk_id, vector):
        with self._vectors_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"chunk_id": chunk_id, "vector": vector},
                                ensure_ascii=False) + "\n")

    def has_vector(self, chunk_id):  return chunk_id in self._vectors
    def get_vector(self, chunk_id):  return self._vectors[chunk_id]
    def is_uploaded(self, chunk_id): return self._manifest.get(chunk_id, {}).get("uploaded", False)

    def save_vector(self, chunk_id, vector):
        self._vectors[chunk_id] = vector
        self._append_vector(chunk_id, vector)

    def mark_uploaded(self, chunk_ids):
        for cid in chunk_ids:
            self._manifest.setdefault(cid, {})["uploaded"] = True
        self._save_manifest()

    def sync_vectors_to_remote(self):
        """Push embeddings.jsonl to remote blob. Call after a batch completes."""
        self._upload_to_remote(self._vectors_path, _EMBED_CACHE_VECTORS)

    def stats(self):
        return (
            len(self._vectors),
            sum(1 for e in self._manifest.values() if e.get("uploaded")),
        )

    def clear(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self._vectors  = {}
        self._manifest = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"🗑️  Embed cache cleared: {self.cache_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def _embed_pipeline_batch(chunks, client, deployment, cache, global_done, global_total):
    """Embed one pipeline batch in EMBED_BATCH_SIZE sub-batches."""
    to_embed = [c for c in chunks if not cache.has_vector(c.chunk_id)]
    for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
        sub   = to_embed[i : i + EMBED_BATCH_SIZE]
        texts = [c.content for c in sub]
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                resp    = client.embeddings.create(input=texts, model=deployment)
                vectors = [item.embedding for item in resp.data]
                for chunk, vec in zip(sub, vectors):
                    cache.save_vector(chunk.chunk_id, vec)
                # time.sleep(20)
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err or "RateLimitReached" in err:
                    wait_m = re.search(r'retry after (\d+) second', err, re.IGNORECASE)
                    wait   = int(wait_m.group(1)) if wait_m else (15 * 2 ** attempt)
                    print(f"\n    ⚠️  Rate-limited — waiting {wait}s "
                          f"(attempt {attempt+1}/{EMBED_MAX_RETRIES})")
                    # time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(
                f"Embedding failed after {EMBED_MAX_RETRIES} retries at sub-batch {i}"
            )
        done = global_done + min(i + EMBED_BATCH_SIZE, len(to_embed))
        print(f"    Embedded {done}/{global_total} …", end="\r")
    return {c.chunk_id: cache.get_vector(c.chunk_id) for c in chunks}


# ─────────────────────────────────────────────────────────────────────────────
# Document builder
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_doc(chunk: Chunk, vector: list[float]) -> dict:
    return {
        "chunk_id":             chunk.chunk_id,
        "source":               chunk.source,
        "source_pdf_url":       chunk.source_pdf_url,
        "source_pdf_name":      chunk.source_pdf_name,
        "chunk_index":          chunk.chunk_index,
        "page_start":           chunk.page_start or 0,
        "page_end":             chunk.page_end   or 0,
        "heading_level":        chunk.heading_level,
        "topic":                chunk.topic,
        "subtopic":             chunk.subtopic,
        "section_title":        chunk.section_title,
        "section":              chunk.section,
        "region":               chunk.region,
        "energy_utility_name":  chunk.energy_utility_name,
        "content":              chunk.content,
        "content_type":         chunk.content_type,
        "has_table":            chunk.has_table,
        "has_figure":           chunk.has_figure,
        "is_table":             chunk.is_table,
        "is_figure":            chunk.is_figure,
        "table_count":          chunk.table_count,
        "figure_count":         chunk.figure_count,
        "metadata_json":        json.dumps(chunk.metadata, ensure_ascii=False),
        "content_vector":       vector,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

def _upload_pipeline_batch(chunks, vectors, search_client, cache, batch_num):
    """Upload one pipeline batch in UPLOAD_BATCH_SIZE sub-batches."""
    pending    = [c for c in chunks if not cache.is_uploaded(c.chunk_id)]
    total_ok   = 0
    total_fail = 0
    for i in range(0, len(pending), UPLOAD_BATCH_SIZE):
        sub     = pending[i : i + UPLOAD_BATCH_SIZE]
        docs    = [_chunk_to_doc(c, vectors[c.chunk_id]) for c in sub]
        results = search_client.upload_documents(documents=docs)
        ok      = [c for c, r in zip(sub, results) if r.succeeded]
        fail    = [c for c, r in zip(sub, results) if not r.succeeded]
        if ok:
            cache.mark_uploaded([c.chunk_id for c in ok])
        total_ok   += len(ok)
        total_fail += len(fail)
        sub_n = i // UPLOAD_BATCH_SIZE + 1
        print(f"    Upload sub-batch {sub_n}: {len(ok)}/{len(sub)} ok"
              + (f"  ⚠️  {len(fail)} failed" if fail else ""))
    return total_ok, total_fail


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(chunks, search_client, openai_client, embed_deploy, cache):
    total    = len(chunks)
    n_rounds = (total + PIPELINE_BATCH_SIZE - 1) // PIPELINE_BATCH_SIZE

    already_emb = sum(1 for c in chunks if cache.has_vector(c.chunk_id))
    already_upl = sum(1 for c in chunks if cache.is_uploaded(c.chunk_id))

    print(f"\n{'─'*60}")
    print(f"  Total chunks      : {total:>6}")
    print(f"  Need embedding    : {total - already_emb:>6}  ({already_emb} cached)")
    print(f"  Need uploading    : {total - already_upl:>6}  ({already_upl} cached)")
    print(f"  Pipeline batches  : {n_rounds:>6}  (size {PIPELINE_BATCH_SIZE})")
    print(f"{'─'*60}\n")

    if already_emb == total and already_upl == total:
        print("✅  Nothing to do — all chunks already embedded and uploaded.")
        return

    embed_done = already_emb
    total_ok   = 0
    total_fail = 0

    for batch_num, start in enumerate(range(0, total, PIPELINE_BATCH_SIZE), start=1):
        batch = chunks[start : start + PIPELINE_BATCH_SIZE]
        end   = min(start + PIPELINE_BATCH_SIZE, total)

        if all(cache.is_uploaded(c.chunk_id) for c in batch):
            print(f"  ⚡  Batch {batch_num}/{n_rounds} [{start+1}–{end}] already uploaded — skip")
            continue

        new_embeds  = sum(1 for c in batch if not cache.has_vector(c.chunk_id))
        new_uploads = sum(1 for c in batch if not cache.is_uploaded(c.chunk_id))
        print(f"\n  ▶  Batch {batch_num}/{n_rounds}  [chunks {start+1}–{end} of {total}]"
              f"  |  embed: {new_embeds} new  |  upload: {new_uploads} pending")

        # 1. Embed
        vectors = _embed_pipeline_batch(
            batch, openai_client, embed_deploy, cache,
            global_done=embed_done, global_total=total,
        )
        if new_embeds:
            print()  # newline after \r progress
        embed_done += new_embeds

        # 2. Upload
        ok, fail = _upload_pipeline_batch(batch, vectors, search_client, cache, batch_num)
        total_ok   += ok
        total_fail += fail

    _, uploaded_total = cache.stats()
    print(f"\n{'─'*60}")
    print(f"✅  Done — {uploaded_total}/{total} chunks in '{INDEX_NAME}'")
    if total_fail:
        print(f"  ⚠️  {total_fail} chunk(s) failed — re-run to retry")
    print(f"{'─'*60}\n")

    # Sync embeddings.jsonl to remote after pipeline completes
    cache.sync_vectors_to_remote()


# ─────────────────────────────────────────────────────────────────────────────
# Index management helpers
# ─────────────────────────────────────────────────────────────────────────────

def delete_index(index_client: SearchIndexClient) -> None:
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"🗑️  Index '{INDEX_NAME}' deleted.")
    except Exception as exc:
        if "ResourceNotFound" in str(exc) or "404" in str(exc):
            print(f"ℹ️  Index '{INDEX_NAME}' did not exist.")
        else:
            raise


def index_exists(index_client: SearchIndexClient) -> bool:
    try:
        index_client.get_index(INDEX_NAME)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_indexer(
    documents_folder:  str,
    recreate:          bool = False,
    clear_embed_cache: bool = False,
) -> None:
    index_client, search_client, openai_client, embed_deploy = _get_clients()

    cache_dir   = Path(
        os.environ.get("EMBED_CACHE_DIR", "") or
        Path(documents_folder) / ".embed_cache"
    )
    embed_cache = EmbedCache(cache_dir)

    if clear_embed_cache:
        embed_cache.clear()

    if recreate:
        print("♻️  Dropping existing index …")
        delete_index(index_client)
    elif index_exists(index_client):
        print(f"ℹ️  Index '{INDEX_NAME}' exists (pass --recreate to rebuild).")

    create_index(index_client)

    chunks = chunk_directory(documents_folder)
    if not chunks:
        print("⚠️  No chunks produced.")
        return

    tables  = sum(1 for c in chunks if c.has_table)
    figures = sum(1 for c in chunks if c.has_figure)
    print(f"   {len(chunks)} chunks  |  {tables} with tables  |  {figures} with figures")

    # Quick summary of utilities found
    utilities = sorted({c.energy_utility_name for c in chunks if c.energy_utility_name})
    print(f"   Utilities: {utilities}")

    run_pipeline(chunks, search_client, openai_client, embed_deploy, embed_cache)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    args             = sys.argv[1:]
    recreate_flag    = "--recreate"    in args
    clear_embed_flag = "--clear-embed" in args
    args             = [a for a in args if not a.startswith("--")]
    folder           = args[0] if args else "./documents"
    run_indexer(folder, recreate=recreate_flag, clear_embed_cache=clear_embed_flag)