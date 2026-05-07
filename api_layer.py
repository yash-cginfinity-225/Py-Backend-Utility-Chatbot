"""
api_layer.py  —  API / Controller Layer (FastAPI)
---------------------------------------------------
Exposes REST endpoints for the React frontend.

Run:
  uvicorn api_layer:app --reload --port 8000
"""

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import logging
import re
from pathlib import Path
import json
from typing import List, Optional
from blob_handler import (
    create_client as blob_create_client,
    download_blob as blob_download_blob,
    list_clients as blob_list_clients,
    list_all_files as blob_list_all_files,
    upload_file as blob_upload_file,
)
from process_files import run_for_files
from service_layer import process_chat
from data_layer import get_clients
import shutil
from typing import Optional
from background_layer import upload_json_blob, backup_session_blobs_to_db, download_json_blob, upsert_feedback, _get_sql_connection
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    scheduler = BackgroundScheduler(timezone=ZoneInfo("America/Chicago"))
    scheduler.add_job(backup_session_blobs_to_db, "cron", hour=2, minute=20)
    # scheduler.add_job(backup_session_blobs_to_db, "interval", minutes=3)
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)

app = FastAPI(
    title="Utility Chatbot API",
    description="RAG backend for the NY Utility Regulatory Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    serial: str | None = None
    top_k: int = Field(default=15, ge=3, le=20)
    top_n_rerank: int = Field(default=15, ge=3, le=15)
    enable_refinement: bool = True
    enable_reranking: bool = True
    filter_content_type: str | None = None
    filter_source: str | None = None


class SourceHit(BaseModel):
    chunk_id: str | None = None
    source: str | None = None
    source_pdf_url: str | None = None
    source_pdf_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    section: str | None = None
    topic: str | None = None
    subtopic: str | None = None
    content: str | None = None
    content_type: str | None = None
    is_table: bool | None = None
    is_figure: bool | None = None
    score: float | None = Field(None, alias="_score")
    rerank_score: float | None = Field(None, alias="_rerank_score")
    caption: str | None = Field(None, alias="_caption")

    model_config = {"populate_by_name": True}


class ChatResponse(BaseModel):
    answer: str
    refined_query: str
    original_query: str
    hits: list[dict]
    serial: str | None = None


class CreateClientRequest(BaseModel):
    name: str


class CreateClientResponse(BaseModel):
    message: str
    client: str


class UploadResponse(BaseModel):
    message: str
    client: str
    files: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Sessions persistence (simple file store)
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS_FILE = Path(__file__).parent / "sessions.json"


def _load_sessions():
    if not SESSIONS_FILE.exists():
        return []
    try:
        raw = SESSIONS_FILE.read_bytes()
        if not raw:
            return []
        last_br = raw.rfind(b']')
        if last_br != -1:
            raw = raw[: last_br + 1]
        text = raw.decode("utf-8", errors="replace")
        m = re.search(r"\[.*\]", text, re.DOTALL)
        json_text = m.group(0).strip() if m else text.strip()

        if not json_text:
            return []

        return json.loads(json_text)
    except Exception:
        logger.exception("Failed to read sessions file")
        return []


def _save_sessions(sessions):
    tmp = SESSIONS_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
        tmp.replace(SESSIONS_FILE)
    except Exception:
        logger.exception("Failed to write sessions file")
        raise HTTPException(status_code=500, detail="Failed to save sessions")


@app.get("/sessions")
def get_sessions():
    return _load_sessions()


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    sessions = _load_sessions()
    for s in sessions:
        if s.get("id") == session_id:
            return s
    raise HTTPException(status_code=404, detail="Session not found")


@app.put("/sessions/{session_id}")
def put_session(session_id: str, session: dict):
    """Create or update a single session (file-backed simple store)."""
    sessions = _load_sessions()
    session["id"] = session_id
    for i, s in enumerate(sessions):
        if s.get("id") == session_id:
            sessions[i] = session
            _save_sessions(sessions)
            return {"ok": True, "session": session}
    sessions.append(session)
    _save_sessions(sessions)
    return {"ok": True, "session": session}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Client / upload endpoints ───────────────────────────

@app.get("/clients", response_model=List[str])
def get_client_list():
    """Return all existing client names."""
    try:
        return blob_list_clients()
    except Exception as exc:
        logger.exception("Failed to list clients")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/clients", response_model=CreateClientResponse)
def post_create_client(body: CreateClientRequest):
    """Create sub-directories for a new client in both Uploaded_Files/ and Markdown_Files/."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Client name must not be empty.")
    try:
        blob_create_client(name)
        return CreateClientResponse(message="Client created", client=name)
    except Exception as exc:
        logger.exception("Failed to create client")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", response_model=UploadResponse)
async def post_upload(
    background_tasks: BackgroundTasks,
    client_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload one or more files for client_name and kick off processing."""
    client_name = client_name.strip()
    if not client_name:
        raise HTTPException(status_code=400, detail="client_name is required.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    uploaded: list[str] = []
    try:
        for f in files:
            blob_upload_file(client_name, f.filename, f.file)
            uploaded.append(f.filename)
            logger.info("Uploaded %s → %s", f.filename, client_name)
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc))

    background_tasks.add_task(run_for_files, client_name, uploaded)
    logger.info("Processing queued for %d file(s) → %s", len(uploaded), client_name)

    return UploadResponse(
        message=f"{len(uploaded)} file(s) uploaded and processing started",
        client=client_name,
        files=uploaded,
    )


@app.get("/clients/{client_name}/files", response_model=List[str])
def get_client_files(client_name: str):
    """Return display names of all uploaded files for a client."""
    try:
        return blob_list_all_files(client_name)
    except Exception as exc:
        logger.exception("Failed to list client files")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/clients/{client_name}/files/{filename}/view")
def view_client_file(client_name: str, filename: str):
    """Download a file for inline viewing. Tries raw upload path first, then processed (read-) path."""
    candidates = [
        f"{client_name}/{filename}",
        f"{client_name}/read-{filename}",
    ]
    for blob_path in candidates:
        try:
            data = blob_download_blob(blob_path)
            content_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
            return Response(
                content=data,
                media_type=content_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )
        except Exception:
            continue
    raise HTTPException(status_code=404, detail=f"File not found: {filename}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint.  Runs the full pipeline:
    refine query → hybrid search → rerank → generate answer.
    """
    try:
        history_dicts = [msg.model_dump() for msg in req.history]

        def _normalize_filter(val: str | None) -> str | None:
            if not val:
                return None
            s = str(val).strip()
            if not s:
                return None
            if s.lower() in {"string", "unknown", "placeholder", "none", "null"}:
                logger.debug("Ignoring placeholder/invalid filter value: %s", s)
                return None
            return s

        filter_content_type = _normalize_filter(req.filter_content_type)
        filter_source = _normalize_filter(req.filter_source)

        result = process_chat(
            question=req.question,
            history=history_dicts,
            top_k=15,
            top_n_rerank=15,
            enable_refinement=False,
            enable_reranking=True,
            filter_content_type=None,
            filter_source=filter_source,
        )

        def sanitize_value(v):
            try:
                if hasattr(v, "text"):
                    return v.text
                if isinstance(v, dict):
                    return {k: sanitize_value(val) for k, val in v.items()}
                if isinstance(v, list):
                    return [sanitize_value(x) for x in v]
                if isinstance(v, (str, int, float, bool)) or v is None:
                    return v
                return str(v)
            except Exception:
                return str(v)

        sanitized_hits = [sanitize_value(h) for h in result.get("hits", [])]

        return ChatResponse(
            answer=result["answer"],
            refined_query=result["refined_query"],
            original_query=result["original_query"],
            hits=sanitized_hits,
            serial=req.serial,
        )

    except RuntimeError as e:
        logger.exception("Service error in /chat")
        # avoid leaking internal details to clients in production
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")


# ─────────────────────────────────────────────────────────────────────────────
# Per-session records  (one JSON file per browser window / session)
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timezone

RECORDS_DIR = Path(__file__).parent / "records"
RECORDS_MEM: dict[str, dict] = {}


def clear_records_dir(records_dir: Optional[str | Path] = None) -> int:
    """No-op clear function to preserve all records.

    Deletion of files is disabled to ensure session/record files are never
    removed by the application. This keeps existing records intact.
    Returns 0 to indicate no entries removed.
    """
    dirp = Path(records_dir) if records_dir else Path(__file__).parent / "records"
    if not dirp.exists():
        logger.debug("Records directory does not exist: %s", dirp)
        return 0
    logger.info("clear_records_dir called but deletion is disabled: %s", dirp)
    return 0


def _session_file(session_id: str) -> Path:
    """Return the path for a session's record file, sanitised to prevent path traversal."""
    safe = re.sub(r"[^\w\-]", "_", session_id)
    return RECORDS_DIR / f"{safe}.json"


def _load_records(session_id: str) -> dict:
    if session_id in RECORDS_MEM:
        return RECORDS_MEM[session_id]
    fpath = _session_file(session_id)
    data = None
    if fpath.exists():
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.exception("Failed to read local records file for session %s", session_id)

    if data is None:
        try:
            safe = re.sub(r"[^\w\-]", "_", session_id)
            blob_name = f"{safe}.json"
            data = download_json_blob(blob_name)
        except Exception:
            logger.exception("Failed to download session blob for %s", session_id)

    if not data:
        return {"session_id": session_id, "messages": []}

    try:
        sess_created = data.pop("created_at", None)
        if sess_created and data.get("messages"):
            first = data["messages"][0]
            if not first.get("created_at"):
                first["created_at"] = sess_created

        RECORDS_MEM[session_id] = data
        return data
    except Exception:
        logger.exception("Failed to read records for session %s", session_id)
        return {"session_id": session_id, "messages": []}


def _save_records(session_id: str, data: dict):
    RECORDS_MEM[session_id] = data
    safe = re.sub(r"[^\w\-]", "_", session_id)
    blob_name = f"{safe}.json"
    try:
        uploaded_url = upload_json_blob(blob_name, data)
        logger.info("Uploaded session data %s to blob: %s", blob_name, uploaded_url)
    except Exception:
        logger.exception("Failed to upload session data to blob: %s", session_id)


class MessageRecord(BaseModel):
    session_id: str
    msg_id: str          
    question: str
    answer: str


class FeedbackRequest(BaseModel):
    session_id: str
    serial: str
    feedback: str


@app.post("/record")
def post_record(req: MessageRecord):
    """Append a question/answer pair to the session's record file."""
    data = _load_records(req.session_id)
    now = datetime.now(timezone.utc).isoformat()
    data["messages"].append({
        "msgId": req.msg_id,
        "question": req.question,
        "answer": req.answer,
        "feedback": None,
        "created_at": now,
    })
    _save_records(req.session_id, data)
    return {"ok": True}


@app.post("/feedback")
def post_feedback(req: FeedbackRequest):
    """Attach user feedback (single string) to an existing message record by serial.

    If the message exists, update its `feedback` field and persist the session blob.
    If the message does not exist, append a new message record with the feedback.
    Also persist feedback to the Azure SQL table immediately (update or insert).
    """
    if not req.session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    data = _load_records(req.session_id)
    serial = req.serial or ""
    found = False
    for msg in data["messages"]:
        if msg.get("msgId") == serial:
            msg["feedback"] = req.feedback
            found = True
            break

    if not found:
        new_msg_id = serial or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        new_msg = {
            "msgId": new_msg_id,
            "question": "",
            "answer": "",
            "feedback": req.feedback,
            "created_at": now,
        }
        data.setdefault("messages", []).append(new_msg)
        serial = new_msg_id

    _save_records(req.session_id, data)

    try:
        ok = upsert_feedback(req.session_id, serial, req.feedback)
        if not ok:
            logger.warning("Failed to persist feedback to DB for %s/%s", req.session_id, serial)
    except Exception:
        logger.exception("Error while persisting feedback to DB for %s/%s", req.session_id, serial)

    return {"ok": True, "serial": serial}


# ─────────────────────────────────────────────────────────────────────────────
# FAQ  —  most frequently asked questions
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/faq")
def get_faq(limit: int = 50):
    """Return the top questions by occurrence count with their most recent answer."""
    try:
        conn = _get_sql_connection()
        cursor = conn.cursor()
        sql = """
            SELECT TOP (?) question, answer
            FROM (
                SELECT question, answer,
                       ROW_NUMBER() OVER (PARTITION BY question ORDER BY created_at DESC) AS rn,
                       COUNT(*) OVER (PARTITION BY question) AS cnt
                FROM [utility].[SessionData]
                WHERE question IS NOT NULL AND question != ''
                  AND answer  IS NOT NULL AND answer  != ''
            ) q
            WHERE rn = 1
            ORDER BY cnt DESC
        """
        cursor.execute(sql, limit)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{"question": r[0], "answer": r[1]} for r in rows]
    except Exception:
        logger.exception("Failed to fetch FAQ data")
        raise HTTPException(status_code=500, detail="Failed to fetch FAQ data")