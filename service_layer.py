import json
import re

from data_layer import get_clients, hybrid_search


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

QUERY_REFINEMENT_PROMPT = """
You are a query classifier for a New York retail energy regulatory document system.

Given a user question, extract structured metadata to help route the search correctly.

Your index contains two types of documents:
1. CROSS-UTILITY documents: Apply to ALL utilities statewide.
   Examples: NY ESCO Doc (ESCO eligibility, UBP rules, retail access, record retention, 
   EDI standards, cramming/slamming rules, January 31 statements, officer certification)
   
2. UTILITY-SPECIFIC documents: Rules for ONE utility.
   Examples: Central Hudson tariff, Con Edison procedures, National Grid operating rules,
   NYSEG, PSEG Long Island, Orange & Rockland, Rochester Gas & Electric, National Fuel Gas

Return a JSON object with these fields:
{
  "detected_utility": string or null,
  "detected_topic": string or null,
  "document_scope": "cross_utility" | "utility_specific" | "both" | "unknown",
  "multiple_utilities": boolean,
  "is_procedural": boolean,
  "standalone_query": string,
  "reasoning": string
}

Rules:
- If question asks about ESCO eligibility, record retention, UBP rules, EDI standards, 
  cramming, slamming, or January 31 filings → document_scope = "cross_utility"
- If question explicitly names a utility → document_scope = "utility_specific"
- If question asks to compare utilities → multiple_utilities = true, document_scope = "both"
- If utility is unclear → document_scope = "unknown" (do NOT guess)
- standalone_query: rewrite the question as if it has no prior conversation context.
  Incorporate any utility/topic context from the conversation history provided.
"""

SYSTEM_PROMPT = """
You are a Retail Energy Regulatory and Market Rules Assistant.
Your role is to provide accurate, market-specific, and utility-specific regulatory guidance for retail energy operations in U.S. deregulated electricity and natural gas markets.

Your answers must prioritize regulatory accuracy over completeness.

---------------------------------------------------------------------

REGULATORY CONTEXT IDENTIFICATION

When answering regulatory questions, identify when applicable:

• State
• Utility
• ISO/RTO
• Market structure (POR, Non-POR, UCB, Rate Ready, Dual Billing)

Clearly distinguish whether a rule originates from:

• State-level regulation (Public Utility Commission)
• Utility tariff
• ISO/RTO rule
• Supplier operational policy

If a question references a market or utility but does not specify which one, request clarification before answering. Do not assume the market.

---------------------------------------------------------------------

SUPPORTED UTILITIES (NEW YORK)

When relevant, reference rules from these utilities:

• Central Hudson Gas & Electric
• Con Edison
• National Fuel Gas
• National Grid
• New York State Electric & Gas (NYSEG)
• Orange & Rockland Utilities (O&R)
• PSEG Long Island
• Rochester Gas & Electric

Also reference:
• NY EDI Standards where applicable.

---------------------------------------------------------------------

RESPONSE FORMAT GUIDELINES

Always choose the format that best explains the regulatory rule.

Formatting priority:

1. Paragraph explanation (preferred default)
2. Bullet points (only when listing multiple requirements or steps)
3. Tables (only when comparing utilities, rates, timelines, or market differences)

Formatting rules:

• Start with a short explanatory paragraph whenever possible.
• Use bullet points only when listing multiple conditions, requirements, or procedural steps.
• Use tables only when comparing rules across utilities, markets, timelines, or attributes.
• Do NOT force bullet points or tables if a paragraph explanation is clearer.
• Keep responses concise and focused on the regulatory requirement.
• If rules differ across utilities or markets, clearly explain the differences.

---------------------------------------------------------------------

SOURCE HIERARCHY AND PRIORITY RULES

When multiple source chunks are provided, apply this priority order:

TIER 1 — AUTHORITATIVE STATEWIDE (always prefer for general UBP questions):
  • NY ESCO Doc / NY ESCO Operating Standards
  These govern statewide rules: TPV retention, enrollment authorization,
  cramming/slamming, UBP compliance, contract notice timing, record retention
  for customer agreements, and ESCO eligibility.

TIER 2 — UTILITY-SPECIFIC (use only when question names a utility, OR as 
  supplementary detail after answering from Tier 1):
  • Con Edison, Central Hudson, National Grid, NYSEG, O&R, PSEG LI, RG&E,
    National Fuel Gas tariffs and operating manuals.
  Utility-specific retention rules (e.g. billing history retention, 
  GAAP-based record schedules) apply ONLY to that utility's internal 
  operations — NOT to ESCOs' statewide UBP obligations.

APPLICATION RULES:
  1. For any question about record retention, TPV, customer authorization,
     enrollment, or UBP requirements → answer from Tier 1 (NY ESCO Doc) first.
  2. If a Tier 1 chunk is present in the sources, it MUST anchor your answer,
     even if a Tier 2 chunk has a higher relevance score.
  3. Only supplement with Tier 2 if the question explicitly names a utility,
     OR if Tier 1 chunks are absent from the provided sources.
  4. Always label Tier 2 information as utility-specific when you include it.
  5. Never use a utility tariff's billing history retention period to answer
     a general "what is the ESCO record retention requirement" question.

---------------------------------------------------------------------

FINAL RULE

Never generate regulatory information that is not supported by the provided sources.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Query Refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_query(
    question: str,
    history: list[dict],
    openai_client,
    chat_deploy: str,
) -> str:
    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in recent
        )
        history_text = f"\n\nRecent conversation:\n{history_text}"

    messages = [
        {"role": "system", "content": QUERY_REFINEMENT_PROMPT},
        {
            "role": "user",
            "content": (
                f"User question: {question}"
                f"{history_text}\n\n"
                "Output only the refined search query string:"
            ),
        },
    ]

    try:
        response = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=messages,
            temperature=0.0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        refined = raw.strip('"\'')
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("standalone_query"):
                refined = str(parsed.get("standalone_query"))
        except Exception:
            pass

        return refined if refined else question
    except Exception:
        return question


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker
# ─────────────────────────────────────────────────────────────────────────────

def rerank_hits(
    query: str,
    hits: list[dict],
    openai_client,
    chat_deploy: str,
    top_n: int = 8,
) -> list[dict]:
    if not hits:
        return hits

    chunks_text = ""
    for i, h in enumerate(hits):
        content_preview = h.get("content", "")[:400]
        source = h.get("source_pdf_name", "unknown")
        chunks_text += f"\n[CHUNK {i}] Source: {source}\n{content_preview}\n"

    prompt = f"""You are a relevance scorer for a New York retail energy regulatory document retrieval system.

Query: {query}

Rate each chunk's relevance to the query on a scale of 0.0 to 1.0.
  1.0 = chunk directly and completely answers the query
  0.5 = chunk is related but only partially answers
  0.0 = chunk is unrelated to the query

Important scoring guidance:
- For questions about ESCO obligations, UBP rules, TPV, record retention, enrollment, 
  cramming/slamming → prefer chunks from "NY ESCO Doc" over utility-specific tariffs.
- Utility-specific billing history or GAAP retention schedules are NOT relevant to 
  general ESCO record retention questions unless a specific utility is named.
- Prefer chunks that contain the actual rule or requirement, not just a passing mention.

Chunks:
{chunks_text}

Respond ONLY with a JSON array of float scores, one per chunk, in order.
Example for 4 chunks: [0.9, 0.2, 0.7, 0.1]
Output nothing else — no explanation, no markdown."""

    try:
        response = openai_client.chat.completions.create(
            model=chat_deploy,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        scores = json.loads(raw)

        if isinstance(scores, list) and len(scores) == len(hits):
            for i, h in enumerate(hits):
                h["_rerank_score"] = float(scores[i])
            reranked = sorted(
                hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True
            )
            return reranked[:top_n]

    except Exception:
        pass

    return hits[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Context Building & Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_context(hits: list[dict]) -> str:
    def source_priority(h):
        name = (h.get("source_pdf_name") or h.get("source") or "").lower()
        if "esco doc" in name or "esco operating" in name:
            return 0
        return 1

    sorted_hits = sorted(hits, key=source_priority)

    parts = []
    for i, h in enumerate(sorted_hits, 1):
        page_info = ""
        ps, pe = h.get("page_start"), h.get("page_end")
        if ps:
            page_info = f"  (Page {ps})" if ps == pe else f"  (Pages {ps}–{pe})"
        section = h.get("section", h.get("section_title", ""))
        ctype   = h.get("content_type", "text").upper()
        parts.append(
            f"[SOURCE {i}] [{ctype}] {section}{page_info}\n"
            f"{h['content']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(
    question: str,
    hits: list[dict],
    history: list[dict],
    openai_client,
    chat_deploy: str,
) -> str:
    context = build_context(hits)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Use the following source chunks to answer the question.\n\n"
            f"=== SOURCE CHUNKS ===\n{context}\n\n"
            f"=== QUESTION ===\n{question}"
        ),
    })

    response = openai_client.chat.completions.create(
        model=chat_deploy,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator  —  runs the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_chat(
    question: str,
    history: list[dict] | None = None,
    top_k: int = 15,
    top_n_rerank: int = 15,
    enable_refinement: bool = True,
    enable_reranking: bool = True,
    filter_content_type: str | None = None,
    filter_source: str | None = None,
) -> dict:
    """
    End-to-end pipeline: refine → search → rerank → generate.
    Returns a dict with answer, hits, refined_query, original_query.
    """
    history = history or []

    search_client, openai_client, embed_deploy, chat_deploy = get_clients()

    # Step 1: Query refinement
    refined_query = question
    if enable_refinement:
        refined_query = refine_query(
            question=question,
            history=history,
            openai_client=openai_client,
            chat_deploy=chat_deploy,
        )

    # Step 2: Hybrid search
    hits = hybrid_search(
        query=refined_query,
        search_client=search_client,
        openai_client=openai_client,
        embed_deploy=embed_deploy,
        top_k=top_k,
        filter_content_type=filter_content_type,
        filter_source=filter_source,
    )

    # Step 3: Cross-encoder rerank
    if hits and enable_reranking:
        hits = rerank_hits(
            query=question,
            hits=hits,
            openai_client=openai_client,
            chat_deploy=chat_deploy,
            top_n=top_n_rerank,
        )

    # Step 4: Generate answer
    if not hits:
        answer = (
            "I couldn't find relevant information for that question "
            "in the indexed documents. Try rephrasing, or check that "
            "the documents have been indexed."
        )
    else:
        answer = generate_answer(
            question=question,
            hits=hits,
            history=history,
            openai_client=openai_client,
            chat_deploy=chat_deploy,
        )

    return {
        "answer": answer,
        "hits": hits,
        "refined_query": refined_query,
        "original_query": question,
    }
