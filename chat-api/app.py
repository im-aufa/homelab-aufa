import os
import logging
import threading
import uuid
import time
from typing import List, Optional, Dict, Any
import glob
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from auth import get_current_user, create_token, LOCAL_USERNAME, LOCAL_PASSWORD

# --- Setup ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # stdout only — no file logging in local dev
)
logger = logging.getLogger(__name__)

# --- Ollama / Model Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "qwen3.5:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

# OpenAI SDK pointed at Ollama; api_key must be non-empty but is otherwise ignored by Ollama.
client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

# Chunk size for document ingestion (tokens, via HybridChunker's internal bert tokenizer).
CHUNK_MAX_TOKENS = 512

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- FastAPI Setup ---
app = FastAPI(title="Cerina RAG API")

# CORS: comma-separated allowed origins. Leave ALLOWED_ORIGINS blank to disallow all cross-origin requests.
_raw_origins  = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Background Job State ---
# Keyed by job_id (UUID str). Written by background thread, read by /process/status.
# Simple dict is fine — GIL protects individual dict writes on CPython.
_jobs: Dict[str, Dict[str, Any]] = {}

def _new_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",   # pending | running | done | error
        "folder": None,
        "total": 0,
        "done": 0,
        "failed": 0,
        "current_file": None,
        "results": [],         # per-doc {"file", "chunks", "status", "error"}
        "started_at": None,
        "finished_at": None,
    }
    return job_id

# --- PostgreSQL Helpers ---
def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(DATABASE_URL)

def setup_postgres():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # The vector extension is created by the superuser in db/setup.sql.
        # This call is a safe no-op if it already exists; failures are logged and skipped.
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            logger.warning(
                "Could not CREATE EXTENSION vector — likely already exists or insufficient privileges. "
                "This is expected if setup.sql was run by the superuser. Continuing."
            )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id           SERIAL PRIMARY KEY,
                doc_id       VARCHAR(255),
                doc_name     VARCHAR(255),
                chunk_index  INTEGER,
                text         TEXT,
                embedding    VECTOR(1024),
                filename     VARCHAR(255),
                page_numbers INTEGER[],
                title        VARCHAR(255),
                source_type  VARCHAR(50),
                UNIQUE (doc_id, chunk_index, source_type)
            )
        """)
        conn.commit()
        logger.info("PostgreSQL setup complete.")
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL setup error: {e}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

# --- Embedding Functions ---
# qwen3-embedding is trained asymmetrically:
#   • Documents: embed as plain text — no prefix needed.
#   • Queries:   prepend a task instruction for best retrieval quality (~1-5% gain).
# Ollama does not apply these templates automatically; we handle them client-side.

def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed document chunks as plain text (no prefix). Used at ingestion time."""
    if not texts:
        return []
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [e.embedding for e in response.data]
    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        raise

def embed_query(query: str) -> List[float]:
    """Embed a user query with the required task instruction prefix. Used at search time."""
    prefixed = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        f"Query: {query}"
    )
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[prefixed])
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        raise

# --- Chunk Storage ---
def store_chunks(chunks_data: List[tuple]):
    if not chunks_data:
        logger.info("No chunks to store.")
        return
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO chunks (doc_id, doc_name, chunk_index, text, embedding, filename, page_numbers, title, source_type)
        VALUES %s
        ON CONFLICT (doc_id, chunk_index, source_type) DO UPDATE SET
            doc_name     = EXCLUDED.doc_name,
            text         = EXCLUDED.text,
            embedding    = EXCLUDED.embedding,
            filename     = EXCLUDED.filename,
            page_numbers = EXCLUDED.page_numbers,
            title        = EXCLUDED.title;
        """
        execute_values(cursor, insert_query, chunks_data, template="(%s, %s, %s, %s, %s::vector, %s, %s, %s, %s)")
        conn.commit()
        logger.info(f"Stored/updated {len(chunks_data)} chunks.")
    except psycopg2.Error as e:
        logger.error(f"Error storing chunks: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# --- Document Processing (local folder only) ---
def process_documents(local_folder_path: str, job_id: str):
    """
    Processes all PDFs in local_folder_path.
    Updates job state in _jobs[job_id] as each document completes.
    Commits chunks to Postgres per-document — a failure on one doc does not
    roll back documents that already succeeded.
    """
    job = _jobs[job_id]
    job["status"] = "running"
    job["folder"] = local_folder_path
    job["started_at"] = time.time()

    if not local_folder_path or not os.path.isdir(local_folder_path):
        job["status"] = "error"
        job["finished_at"] = time.time()
        raise ValueError(f"Invalid local folder path: {local_folder_path}")

    pdf_files = glob.glob(os.path.join(local_folder_path, "*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {local_folder_path}.")
        job["status"] = "done"
        job["finished_at"] = time.time()
        return

    job["total"] = len(pdf_files)

    converter = DocumentConverter()
    chunker   = HybridChunker(tokenizer="bert-base-uncased", max_tokens=CHUNK_MAX_TOKENS, merge_peers=True)

    for pdf_path in pdf_files:
        doc_name = os.path.basename(pdf_path)
        doc_id   = doc_name
        job["current_file"] = doc_name
        logger.info(f"[{job_id}] Processing ({job['done'] + 1}/{job['total']}): {doc_name}")

        try:
            result     = converter.convert(pdf_path)
            document   = result.document
            chunk_iter = chunker.chunk(dl_doc=document)
            doc_chunks = list(chunk_iter)

            if not doc_chunks:
                logger.warning(f"[{job_id}] No chunks generated for: {doc_name}")
                job["results"].append({"file": doc_name, "chunks": 0, "status": "skipped", "error": "no chunks"})
                job["done"] += 1
                continue

            chunk_texts = [chunk.text for chunk in doc_chunks]
            embeddings  = embed_documents(chunk_texts)

            chunks_to_store = []
            for i, chunk in enumerate(doc_chunks):
                page_numbers = sorted(list(set(
                    prov.page_no for item in chunk.meta.doc_items
                    for prov in item.prov if prov.page_no is not None
                ))) or None
                title = chunk.meta.headings[0] if chunk.meta.headings else None
                chunks_to_store.append((
                    doc_id, doc_name, i, chunk.text, embeddings[i],
                    doc_name, page_numbers, title, "local"
                ))

            # Commit this doc immediately — independent of other docs.
            store_chunks(chunks_to_store)
            job["results"].append({"file": doc_name, "chunks": len(chunks_to_store), "status": "ok"})
            job["done"] += 1
            logger.info(f"[{job_id}] ✅ Done: {doc_name} ({len(chunks_to_store)} chunks)")

        except Exception as e:
            logger.error(f"[{job_id}] ❌ Failed: {doc_name} — {e}")
            job["results"].append({"file": doc_name, "chunks": 0, "status": "error", "error": str(e)})
            job["failed"] += 1
            job["done"] += 1  # still counts as processed so total/done math stays consistent

    job["current_file"] = None
    job["status"] = "done" if job["failed"] == 0 else "done_with_errors"
    job["finished_at"] = time.time()
    logger.info(f"[{job_id}] Ingestion complete. {job['done']} processed, {job['failed']} failed.")

# --- RAG Search & Query ---
def search_chunks(query: str, n_results: int = 5) -> List[dict]:
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query_embedding = embed_query(query)
        cursor.execute("""
            SELECT doc_id, doc_name, chunk_index, text, filename, page_numbers, title, source_type,
                   (embedding <=> %s::vector) as distance
            FROM chunks
            ORDER BY distance ASC
            LIMIT %s
        """, (query_embedding, n_results))
        results = [
            {
                "doc_id": row[0], "doc_name": row[1], "chunk_index": row[2],
                "text": row[3], "filename": row[4], "page_numbers": row[5],
                "title": row[6], "source_type": row[7], "distance": row[8]
            }
            for row in cursor.fetchall()
        ]
        return results
    except psycopg2.Error as e:
        logger.error(f"Database error during chunk search: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()

def query_rag(query: str, n_results: int = 5) -> str:
    try:
        relevant_chunks = search_chunks(query, n_results)
        if not relevant_chunks:
            return (
                "I'm Cerina, and I'm here to help! I wasn't able to find anything in the documents "
                "that directly answers your question. Feel free to ask something else — I'll do my best."
            )
        context = []
        for res in relevant_chunks:
            source_parts = []
            if res.get("doc_name"):
                source_parts.append(f"Document: '{res['doc_name']}' ({res.get('source_type', 'unknown')})")
            if res.get("page_numbers"):
                source_parts.append(f"Pages: {', '.join(map(str, res['page_numbers']))}")
            if res.get("title"):
                source_parts.append(f"Section: '{res['title']}'")
            source_info = " | ".join(filter(None, source_parts))
            context.append(f"Context from {source_info}:\n{res['text']}\n---")
        context_str = "\n\n".join(context)

        # DRAFT system prompt — review tone/scope in docs/decisions/local-embeddings.md before going live.
        system_message = (
            "You are Cerina, a warm and caring health companion. You're here to help people understand "
            "their bodies with kindness, clarity, and honesty — especially around menstrual health and cervical wellness.\n\n"
            "Answer questions based only on the information provided in the context below. If the context "
            "doesn't cover something clearly, say so gently — never guess or fill in gaps with assumptions. "
            "When you have useful information, mention where it comes from (document name, page, or section) "
            "so the person can explore further if they'd like.\n\n"
            "Be warm, human, and approachable. Avoid clinical distance. You're talking with a real person "
            "who deserves a thoughtful, caring answer."
        )

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": f"User Question: {query}\n\nProvided Context:\n{context_str}"}
            ],
            max_tokens=700,
            temperature=0.7,
            extra_body={"think": False}  # Disable qwen3 thinking/reasoning mode — avoids ~3 min latency per response
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        return "I'm Cerina, your health companion. Something went wrong on my end — please try again in a moment."

# --- Pydantic Models ---
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class QueryRequest(BaseModel):
    query: str
    n_results: int = Field(5, ge=1, le=20)

class QueryResponse(BaseModel):
    response: str
    retrieved_chunks: Optional[List[dict]] = None

class ProcessRequest(BaseModel):
    local_folder_path: str = "/app/documents"

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None

class ProcessStatusResponse(BaseModel):
    job_id: str
    status: str          # pending | running | done | done_with_errors | error
    folder: Optional[str] = None
    total: int = 0
    done: int = 0
    failed: int = 0
    current_file: Optional[str] = None
    results: List[dict] = []
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    return {"message": "Cerina RAG API is running."}

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Simple local login — returns a signed token on correct credentials."""
    if request.username != LOCAL_USERNAME or request.password != LOCAL_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(request.username)
    return LoginResponse(access_token=token)

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest, user: dict = Depends(get_current_user)):
    logger.info(f"Received query from '{user.get('sub')}': '{request.query}'")
    response_text = query_rag(request.query, request.n_results)
    return QueryResponse(response=response_text)

@app.post("/process", response_model=ProcessResponse)
async def process_endpoint(request: ProcessRequest, user: dict = Depends(get_current_user)):
    """
    Kicks off document ingestion as a background thread and returns immediately.
    Poll /process/status/{job_id} to track progress.
    """
    job_id = _new_job()
    _jobs[job_id]["folder"] = request.local_folder_path

    def _run():
        try:
            process_documents(local_folder_path=request.local_folder_path, job_id=job_id)
        except Exception as e:
            logger.error(f"[{job_id}] Unhandled error in background thread: {e}", exc_info=True)
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["finished_at"] = time.time()

    thread = threading.Thread(target=_run, daemon=True, name=f"ingest-{job_id[:8]}")
    thread.start()

    logger.info(f"[{job_id}] Ingestion job started for: {request.local_folder_path}")
    return ProcessResponse(
        job_id=job_id,
        status="started",
        message=f"Processing '{request.local_folder_path}' in the background. Poll /process/status/{job_id} for progress.",
    )

@app.get("/process/status/{job_id}", response_model=ProcessStatusResponse)
async def process_status_endpoint(job_id: str, user: dict = Depends(get_current_user)):
    """Returns the current state of a background ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return ProcessStatusResponse(job_id=job_id, **job)

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    try:
        setup_postgres()
        logger.info("PostgreSQL connection verified.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}")