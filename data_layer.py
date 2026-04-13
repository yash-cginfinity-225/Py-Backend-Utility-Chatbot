import os
from typing import Optional

from dotenv import load_dotenv

import logging

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Azure client initialisation
# ─────────────────────────────────────────────────────────────────────────────

_clients_cache: dict = {}


def get_clients():
    if _clients_cache.get("initialised"):
        return (
            _clients_cache["search_client"],
            _clients_cache["openai_client"],
            _clients_cache["embed_deploy"],
            _clients_cache["chat_deploy"],
        )

    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from openai import AzureOpenAI

    endpoint   = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    key        = os.environ.get("AZURE_SEARCH_ADMIN_KEY", "")
    index_name = os.environ.get("SEARCH_INDEX_NAME", "edi-documents")
    aoai_ep    = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    aoai_key   = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if not all([endpoint, key, aoai_ep, aoai_key]):
        raise RuntimeError(
            "Missing required environment variables. "
            "Set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_KEY."
        )

    cred          = AzureKeyCredential(key)
    search_client = SearchClient(endpoint, index_name, cred)
    openai_client = AzureOpenAI(
        azure_endpoint=aoai_ep,
        api_key=aoai_key,
        api_version="2024-02-01",
    )
    embed_deploy = os.environ.get("AZURE_OPENAI_EMBED_DEPLOY", "text-embedding-3-small")
    chat_deploy  = "gpt-4o-mini"

    _clients_cache.update({
        "initialised": True,
        "search_client": search_client,
        "openai_client": openai_client,
        "embed_deploy": embed_deploy,
        "chat_deploy": chat_deploy,
    })

    return search_client, openai_client, embed_deploy, chat_deploy


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_query(text: str, client, deploy: str) -> list[float]:
    resp = client.embeddings.create(input=[text], model=deploy)
    return resp.data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    search_client,
    openai_client,
    embed_deploy: str,
    top_k: int = 15,
    filter_content_type: Optional[str] = None,
    filter_source: Optional[str] = None,
) -> list[dict]:
    from azure.search.documents.models import VectorizedQuery

    vector = embed_query(query, openai_client, embed_deploy)
    vq = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k * 2,
        fields="content_vector",
    )

    filters = []

    allowed_content_types = {"all", "text", "table", "figure", "list"}
    if filter_content_type:
        fct = str(filter_content_type).strip()
        if fct.lower() in allowed_content_types and fct.lower() != "all":
            filters.append(f"content_type eq '{fct.lower()}'")
        else:
            logging.getLogger(__name__).debug(
                "Ignoring invalid content_type filter: %s", fct
            )

    if filter_source:
        fs = str(filter_source).strip()
        if fs.lower() in ("all", ""):
            pass
        elif fs.lower() in ("string", "unknown", "placeholder", "none", "null"):
            logging.getLogger(__name__).debug(
                "Ignoring placeholder/invalid source filter: %s", fs
            )
        else:
            safe = fs.replace("'", "''")
            filters.append(f"source_pdf_name eq '{safe}'")

    odata_filter = " and ".join(filters) if filters else None

    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        search_fields=[
            "section_title",
            "topic",
            "subtopic",
            "content",
        ],
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        query_caption="extractive",
        query_answer="extractive",
        top=top_k,
        filter=odata_filter,
        select=[
            "chunk_id", "source", "source_pdf_url", "source_pdf_name",
            "chunk_index", "page_start", "page_end",
            "section_title", "section", "topic", "subtopic",
            "content", "content_type", "is_table", "is_figure",
            "metadata_json",
        ],
    )

    hits = []
    for r in results:
        d = dict(r)
        captions = r.get("@search.captions", [])
        if captions:
            d["_caption"] = captions[0].text
        d["_score"] = r.get("@search.reranker_score") or r.get("@search.score", 0)
        hits.append(d)

    return hits
