from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timezone

import pyodbc
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def _get_blob_service_client(connection_string: Optional[str] = None) -> BlobServiceClient:
	cs = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
	if not cs:
		raise ValueError(
			"Azure Storage connection string must be provided via `connection_string` "
			"or the AZURE_STORAGE_CONNECTION_STRING environment variable"
		)
	return BlobServiceClient.from_connection_string(cs)

def upload_json_blob(file_name: str, data, connection_string: Optional[str] = None) -> str:
	"""Upload JSON content (dict/list or JSON string) directly to SessionData/<file_name>.

	This avoids requiring a local file on disk. `data` may be a dict/list or
	a JSON string; dict/list values will be serialized to JSON.
	Returns the uploaded blob URL.
	"""
	if isinstance(data, (dict, list)):
		text = json.dumps(data, ensure_ascii=False, indent=2)
	else:
		text = str(data)

	container = os.getenv("AZURE_BLOB_CONTAINER") or os.getenv("AZURE_CONTAINER") or "utility-chatbot"
	storage_account = os.getenv("AZURE_STORAGE_ACCOUNT") or "cgdatateam"

	blob_name = f"SessionData/{file_name}"

	svc = _get_blob_service_client(connection_string)
	container_client = svc.get_container_client(container)
	try:
		container_client.create_container()
	except ResourceExistsError:
		pass
	except Exception:
		logger.debug("create_container() raised an unexpected exception", exc_info=True)

	blob_client = container_client.get_blob_client(blob_name)
	content_settings = ContentSettings(content_type="application/json")
	blob_client.upload_blob(text.encode("utf-8"), overwrite=True, content_settings=content_settings)

	try:
		blob_url = blob_client.url
	except Exception:
		blob_url = f"https://{storage_account}.blob.core.windows.net/{container}/{blob_name}"
	return blob_url


def download_json_blob(file_name: str, connection_string: Optional[str] = None) -> Optional[dict]:
	"""Download a JSON blob from SessionData/<file_name> and return parsed JSON.

	Returns None if the blob does not exist or cannot be parsed.
	"""
	container = os.getenv("AZURE_BLOB_CONTAINER") or os.getenv("AZURE_CONTAINER") or "utility-chatbot"
	svc = _get_blob_service_client(connection_string)
	container_client = svc.get_container_client(container)
	blob_name = f"SessionData/{file_name}"
	try:
		blob_client = container_client.get_blob_client(blob_name)
		stream = blob_client.download_blob()
		raw = stream.readall()
		text = raw.decode("utf-8")
		return json.loads(text)
	except Exception:
		logger.debug("Failed to download/parse blob %s", blob_name, exc_info=True)
		return None


def upsert_feedback(session_id: str, msg_id: str, feedback: str) -> bool:
	"""Update feedback for an existing DB row, or insert a new row if not found.

	Returns True on success, False on error.
	"""
	try:
		conn = _get_sql_connection()
		cursor = conn.cursor()
		sel = "SELECT COUNT(1) FROM [utility].[SessionData] WHERE session_id = ? AND msg_id = ?"
		cursor.execute(sel, session_id, msg_id)
		row = cursor.fetchone()
		exists = bool(row and row[0])

		if exists:
			upd = "UPDATE [utility].[SessionData] SET feedback = ? WHERE session_id = ? AND msg_id = ?"
			cursor.execute(upd, feedback, session_id, msg_id)
		else:
			ins = "INSERT INTO [utility].[SessionData] (session_id, msg_id, question, answer, feedback, created_at) VALUES (?, ?, ?, ?, ?, ?)"
			created_at = datetime.now(timezone.utc).isoformat()
			cursor.execute(ins, session_id, msg_id, "", "", feedback, created_at)

		conn.commit()
		cursor.close()
		conn.close()
		logger.info("Upserted feedback for %s/%s (exists=%s)", session_id, msg_id, exists)
		return True
	except Exception:
		logger.exception("Failed to upsert feedback for %s/%s", session_id, msg_id)
		try:
			cursor.close()
		except Exception:
			pass
		try:
			conn.close()
		except Exception:
			pass
		return False


# ─────────────────────────────────────────────────────────────────────────────
# Midnight backup: Blob JSON → Azure SQL, then delete blobs
# ─────────────────────────────────────────────────────────────────────────────

def _get_sql_connection() -> pyodbc.Connection:
	server = (os.getenv("AZURE_SERVER_NAME") or "").strip()
	database = (os.getenv("AZURE_SQL_DATABASE") or "").strip()
	username = (os.getenv("AZURE_SQL_USERNAME") or "").strip()
	password = (os.getenv("AZURE_SQL_PASSWORD") or "").strip()
	if not all([server, database, username, password]):
		raise ValueError(
			"Missing Azure SQL env vars. Set AZURE_SERVER_NAME, AZURE_SQL_DATABASE, "
			"AZURE_SQL_USERNAME, and AZURE_SQL_PASSWORD."
		)

	try:
		available_drivers = pyodbc.drivers()
	except Exception:
		available_drivers = []

	preferred = None
	for candidate in (
		"ODBC Driver 18 for SQL Server",
		"ODBC Driver 17 for SQL Server",
		"ODBC Driver 13 for SQL Server",
		"SQL Server Native Client 11.0",
	):
		if candidate in available_drivers:
			preferred = candidate
			break

	if not preferred and available_drivers:
		# fallback: pick any driver that looks like a SQL Server driver
		for d in available_drivers:
			if "SQL" in d or "ODBC" in d:
				preferred = d
				break

	if not preferred:
		raise RuntimeError(
			"No suitable ODBC driver found.\n"
			"Install the Microsoft ODBC Driver for SQL Server on this host. "
			"See: https://learn.microsoft.com/sql/connect/odbc/windows/microsoft-odbc-driver-for-sql-server"
		)

	# conn_str = (
	# 	f"DRIVER={{{preferred}}};"
	# 	f"SERVER={server};"
	# 	f"DATABASE={database};"
	# 	f"UID={username};"
	# 	f"PWD={password};"
	# 	"Encrypt=yes;TrustServerCertificate=no;"
	# )
	conn_str = (
        f"DRIVER={{{preferred}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;Login Timeout=60;"
    )

	try:
		conn = pyodbc.connect(conn_str)
		logger.info("Connected to Azure SQL using ODBC driver: %s", preferred)
		return conn
	except pyodbc.InterfaceError as e:
		logger.exception("pyodbc InterfaceError connecting to DB using driver %s", preferred)
		raise
	except Exception:
		logger.exception("Failed to connect to Azure SQL")
		raise


def backup_session_blobs_to_db() -> dict:
	"""Download every JSON blob under SessionData/, insert rows into
	[utility].[SessionData], then delete the blobs.

	Returns a summary dict with counts of rows inserted and blobs deleted.
	"""
	container = os.getenv("AZURE_BLOB_CONTAINER") or os.getenv("AZURE_CONTAINER") or "utility-chatbot"
	prefix = "SessionData/"

	svc = _get_blob_service_client()
	container_client = svc.get_container_client(container)

	blobs = list(container_client.list_blobs(name_starts_with=prefix))
	if not blobs:
		logger.info("No session blobs found under %s – nothing to back up.", prefix)
		return {"rows_inserted": 0, "blobs_deleted": 0}

	conn = _get_sql_connection()
	cursor = conn.cursor()

	insert_sql = """
		INSERT INTO [utility].[SessionData]
			(session_id, msg_id, question, answer, feedback, created_at)
		VALUES (?, ?, ?, ?, ?, ?)
	"""

	rows_inserted = 0
	blobs_deleted = 0

	for blob in blobs:
		blob_client = container_client.get_blob_client(blob.name)
		try:
			raw = blob_client.download_blob().readall()
			data = json.loads(raw)
		except Exception:
			logger.exception("Failed to download/parse blob %s – skipping.", blob.name)
			continue

		session_id = data.get("session_id", "")
		messages = data.get("messages", [])

		inserted_this_blob = 0
		failed_inserts: list[str] = []

		for msg in messages:
			feedback_val = msg.get("feedback")
			if feedback_val is None:
				feedback_str = None
			elif isinstance(feedback_val, str):
				feedback_str = feedback_val
			else:
				feedback_str = json.dumps(feedback_val, ensure_ascii=False)

			created_at_val = msg.get("created_at") or msg.get("createdAt")
			if not created_at_val:
				created_at_val = datetime.now(timezone.utc).isoformat()

			try:
				cursor.execute(
					insert_sql,
					session_id,
					msg.get("msgId", ""),
					msg.get("question", ""),
					msg.get("answer", ""),
					feedback_str,
					created_at_val,
				)
				conn.commit()
				rows_inserted += 1
				inserted_this_blob += 1
			except Exception:
				try:
					conn.rollback()
				except Exception:
					pass
				mid = msg.get("msgId", "<unknown>")
				failed_inserts.append(str(mid))
				logger.exception("Failed to insert message %s in blob %s; continuing.", mid, blob.name)

		try:
			blob_client.delete_blob()
			blobs_deleted += 1
			logger.info(
				"Processed blob %s: inserted=%d failed=%d — blob deleted.",
				blob.name,
				inserted_this_blob,
				len(failed_inserts),
			)
		except Exception:
			logger.exception("Failed to delete blob %s after processing (it may remain).", blob.name)

	cursor.close()
	conn.close()

	logger.info(
		"Backup complete: %d rows inserted, %d blobs deleted.", rows_inserted, blobs_deleted
	)
	return {"rows_inserted": rows_inserted, "blobs_deleted": blobs_deleted}
