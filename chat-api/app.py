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
import asyncio
import httpx
from openai import OpenAI, AsyncOpenAI
import psycopg2
from psycopg2.extras import execute_values
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from auth import get_current_user, create_token, LOCAL_USERNAME, LOCAL_PASSWORD

# --- Setup ---
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # stdout only — no file logging in local dev
)
logger = logging.getLogger(__name__)

# Silence overly verbose third-party loggers so our logs stay readable
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)

# --- Ollama / Model Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "qwen3.5:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

logger.info(f"[CONFIG] OLLAMA_BASE_URL={OLLAMA_BASE_URL}")
logger.info(f"[CONFIG] CHAT_MODEL={CHAT_MODEL}")
logger.info(f"[CONFIG] EMBEDDING_MODEL={EMBEDDING_MODEL}")

# Ollama native API base (strips /v1 suffix if present) — used for think: false support.
OLLAMA_NATIVE_URL = OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1").removesuffix("/v1/")
logger.info(f"[CONFIG] OLLAMA_NATIVE_URL={OLLAMA_NATIVE_URL}")

# Sync client — used by background ingestion thread (can't await there).
client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

# Async client — used by chat endpoint so LLM calls don't block the event loop.
async_client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

# Cosine distance threshold for RAG retrieval.
# Chunks with distance > this value are considered irrelevant and filtered out.
SIMILARITY_THRESHOLD = 0.50

# Chunk size for document ingestion (tokens, via HybridChunker's internal bert tokenizer).
CHUNK_MAX_TOKENS = 512

# --- Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")

# --- FastAPI Setup ---
app = FastAPI(title="Cerina RAG API")

# CORS: comma-separated allowed origins. Leave ALLOWED_ORIGINS blank to disallow all cross-origin requests.
_raw_origins  = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
logger.info(f"[CONFIG] ALLOWED_ORIGINS={allowed_origins or '(none — cross-origin requests blocked)'}")

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
    logger.debug(f"[EMBED] embed_documents: sending {len(texts)} text(s) to Ollama ({EMBEDDING_MODEL})")
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        logger.debug(f"[EMBED] embed_documents: received {len(response.data)} embedding(s)")
        return [e.embedding for e in response.data]
    except Exception as e:
        logger.error(f"[EMBED] embed_documents failed: {type(e).__name__}: {e}")
        raise

async def embed_query(query: str) -> List[float]:
    """Embed a user query with the required task instruction prefix. Used at search time."""
    prefixed = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        f"Query: {query}"
    )
    logger.debug(f"[EMBED] embed_query: sending query to Ollama ({EMBEDDING_MODEL}), len={len(query)}")
    try:
        response = await async_client.embeddings.create(model=EMBEDDING_MODEL, input=[prefixed])
        logger.debug(f"[EMBED] embed_query: embedding received, dim={len(response.data[0].embedding)}")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"[EMBED] embed_query failed: {type(e).__name__}: {e}")
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
async def search_chunks(query: str, n_results: int = 5) -> List[dict]:
    """Embed the query and retrieve the top-N chunks above the similarity threshold."""
    logger.info(f"[RAG] search_chunks: query='{query[:80]}{'...' if len(query) > 80 else ''}', n_results={n_results}")
    try:
        query_embedding = await embed_query(query)
    except Exception:
        logger.error("[RAG] search_chunks: embed_query failed — returning empty results")
        return []  # embed_query already logged the error

    def _db_search():
        conn = None
        try:
            conn = get_connection()
            logger.debug("[RAG] _db_search: DB connection established")
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id, doc_name, chunk_index, text, filename, page_numbers, title, source_type,
                       (embedding <=> %s::vector) as distance
                FROM chunks
                ORDER BY distance ASC
                LIMIT %s
            """, (query_embedding, n_results))
            rows = cursor.fetchall()
            logger.debug(f"[RAG] _db_search: {len(rows)} row(s) returned before threshold filter")
            filtered = [
                {
                    "doc_id": row[0], "doc_name": row[1], "chunk_index": row[2],
                    "text": row[3], "filename": row[4], "page_numbers": row[5],
                    "title": row[6], "source_type": row[7], "distance": row[8]
                }
                for row in rows
                if row[8] <= SIMILARITY_THRESHOLD  # filter irrelevant chunks
            ]
            if rows:
                distances = [round(row[8], 4) for row in rows]
                logger.debug(f"[RAG] _db_search: distances={distances}, threshold={SIMILARITY_THRESHOLD}")
            logger.info(f"[RAG] _db_search: {len(filtered)} chunk(s) passed similarity threshold")
            return filtered
        except psycopg2.Error as e:
            logger.error(f"[RAG] _db_search: DB error: {type(e).__name__}: {e}")
            return []
        finally:
            if conn:
                cursor.close()
                conn.close()

    return await asyncio.to_thread(_db_search)


async def stream_rag(query: str, n_results: int = 5):
    """
    Async generator that yields Server-Sent Events (SSE) for each token
    produced by the LLM.  Format:
        data: {"token": "…"}\n\n   — for each streamed token chunk
        data: {"done": true}\n\n  — once the stream finishes
        data: {"error": "…"}\n\n  — on failure
    """
    logger.info(f"[STREAM] stream_rag: START — query='{query[:80]}{'...' if len(query) > 80 else ''}'")
    try:
        relevant_chunks = await search_chunks(query, n_results)

        if relevant_chunks:
            context_parts = []
            for res in relevant_chunks:
                source_parts = []
                if res.get("doc_name"):
                    source_parts.append(f"Document: '{res['doc_name']}'")
                if res.get("page_numbers"):
                    source_parts.append(f"Pages: {', '.join(map(str, res['page_numbers']))}")
                if res.get("title"):
                    source_parts.append(f"Section: '{res['title']}'")
                source_info = " | ".join(filter(None, source_parts))
                context_parts.append(f"[{source_info}]\n{res['text']}\n---")
            context_str = "\n\n".join(context_parts)

            system_message = (
                "You are Cerina, a warm and caring health companion specialising in menstrual health "
                "and cervical wellness. Relevant document excerpts are provided below.\n\n"
                "Use those excerpts as your primary source. You may supplement with your general medical "
                "knowledge to give a complete answer, but always make clear when you're drawing on the "
                "documents vs. general knowledge. Cite the document name and page when helpful.\n\n"
                "Be warm, human, and approachable — never clinical or cold."
            )
            user_message = f"{query}\n\nRelevant document excerpts:\n{context_str}"
            logger.info(f"[STREAM] Responding with {len(relevant_chunks)} relevant chunk(s).")
        else:
            system_message = (
                "You are Cerina, a warm and caring health companion specialising in menstrual health "
                "and cervical wellness. No specific documents were found for this question.\n\n"
                "Answer using your general knowledge. Be honest about uncertainty. For greetings or "
                "off-topic messages, respond naturally and warmly — you don't have to stay on-topic "
                "for every single message.\n\n"
                "Be warm, human, and approachable — never clinical or cold."
            )
            user_message = query
            logger.info("No relevant chunks found — using general knowledge.")

        # Use Ollama's native /api/chat endpoint directly — the only reliable way to
        # pass think: false across all Ollama versions. The OpenAI-compat layer silently
        # swallows thinking tokens and ignores extra_body think flags on older builds.
        ollama_url = f"{OLLAMA_NATIVE_URL}/api/chat"
        payload = {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            "think": False,
            "stream": True,
            "options": {
                "num_predict": 1024,
                "temperature": 0.7,
            },
        }
        logger.info(f"[STREAM] Calling Ollama native API → POST {ollama_url} (think=false, num_predict=1024)")

        token_count = 0
        thinking_token_count = 0
        async with httpx.AsyncClient(timeout=300.0) as http:
            async with http.stream("POST", ollama_url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error(f"[STREAM] Ollama returned HTTP {resp.status_code}: {body.decode()[:200]}")
                    yield f"data: {json.dumps({'error': f'Ollama error {resp.status_code}'})}\n\n"
                    return

                logger.debug("[STREAM] Ollama stream opened — reading tokens ...")
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.warning(f"[STREAM] Could not parse line: {raw_line[:100]}")
                        continue

                    # Log thinking tokens separately (they appear in data["thinking"])
                    thinking = data.get("thinking") or ""
                    if thinking:
                        thinking_token_count += len(thinking)
                        logger.debug(f"[STREAM] Thinking chunk, cumulative thinking chars={thinking_token_count}")

                    content = (data.get("message") or {}).get("content") or ""
                    if content:
                        token_count += 1
                        yield f"data: {json.dumps({'token': content})}\n\n"

                    if data.get("done"):
                        stats = {k: data.get(k) for k in ("total_duration", "eval_count", "prompt_eval_count") if k in data}
                        logger.info(f"[STREAM] Ollama done signal. tokens_sent={token_count}, thinking_chars={thinking_token_count}, stats={stats}")
                        break

        logger.info(f"[STREAM] stream_rag: DONE — {token_count} token chunk(s) sent")
        yield f"data: {json.dumps({'done': True})}\n\n"


    except Exception as e:
        logger.error(f"[STREAM] stream_rag: UNHANDLED ERROR — {type(e).__name__}: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': 'Something went wrong on my end — please try again in a moment.'})}\n\n"


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
    logger.info(f"[LOGIN] Attempt for username='{request.username}'")
    if request.username != LOCAL_USERNAME or request.password != LOCAL_PASSWORD:
        logger.warning(f"[LOGIN] FAILED for username='{request.username}' — invalid credentials")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(request.username)
    logger.info(f"[LOGIN] SUCCESS for username='{request.username}'")
    return LoginResponse(access_token=token)

@app.post("/chat")
async def chat_endpoint(request: QueryRequest, user: dict = Depends(get_current_user)):
    logger.info(f"[CHAT] Received query from user='{user.get('sub')}': '{request.query[:80]}{'...' if len(request.query) > 80 else ''}'")
    return StreamingResponse(
        stream_rag(request.query, request.n_results),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind a proxy
        },
    )

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
async def _warmup_models():
    """Pre-load both Ollama models so the first user request doesn't cold-start timeout."""
    logger.info("Warming up Ollama models in background...")
    try:
        await async_client.embeddings.create(model=EMBEDDING_MODEL, input=["warmup"])
        logger.info(f"✅ Embedding model '{EMBEDDING_MODEL}' warmed up.")
    except Exception as e:
        logger.warning(f"Embedding warmup failed (non-fatal): {e}")
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            r = await http.post(
                f"{OLLAMA_NATIVE_URL}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "think": False,
                    "stream": False,
                    "options": {"num_predict": 5},
                },
            )
            r.raise_for_status()
        logger.info(f"✅ Chat model '{CHAT_MODEL}' warmed up.")
    except Exception as e:
        logger.warning(f"Chat model warmup failed (non-fatal): {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    try:
        setup_postgres()
        logger.info("PostgreSQL connection verified.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}")
    # Warm up Ollama models in background — doesn't block startup
    asyncio.create_task(_warmup_models())