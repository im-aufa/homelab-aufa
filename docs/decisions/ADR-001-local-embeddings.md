# ADR-001: Switch to Local Embeddings (qwen3-embedding:0.6b)

**Date:** 2026-10-03  
**Status:** Accepted

---

## Context

Cerina's original RAG pipeline used OpenAI's `text-embedding-3-large` model with a vector output of **3072 dimensions**. This required an active OpenAI API key, incurred per-call cost, and sent document content to an external service — a poor fit for a self-hosted homelab setup where an Ollama instance is already running.

The server already had `qwen3-embedding:0.6b` available via Ollama, making a zero-cost, fully local alternative viable.

## Decision

Switch the embedding model from `text-embedding-3-large` (OpenAI) to `qwen3-embedding:0.6b` (Ollama).  
Switch the chat model from `gpt-3.5-turbo` (OpenAI) to `qwen3.5:4b` (Ollama).

## Consequences

### Vector dimension change: 3072 → 1024

`qwen3-embedding:0.6b` outputs **1024-dimensional** vectors by default (with Matryoshka support down to 32).  
`text-embedding-3-large` outputs 3072-dimensional vectors.

**Impact:**
- The `chunks.embedding` column definition changed from `VECTOR(3072)` to `VECTOR(1024)`.
- All existing embeddings are **incompatible and must be regenerated**. The embedding spaces are different models — a 3072-dim OpenAI vector stored in a 1024-dim column would be invalid, and even if truncated, would return garbage similarity scores.
- Re-running `/process` against the source documents is required after deploy. This is not data loss — the source PDFs are untouched.

### Asymmetric prefix convention

`qwen3-embedding` is trained asymmetrically. Queries and documents must be embedded differently:

| Input type | Format |
|---|---|
| Document chunk (ingestion) | Plain text — no prefix |
| User query (search) | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}` |

Ollama does not apply these templates automatically. The prefix logic lives in two separate functions in `app.py`: `embed_documents()` and `embed_query()`. Failing to apply the query prefix results in a 1–5% retrieval quality drop.

### tiktoken removed

`tiktoken` was used only to count tokens for the OpenAI embedding limit guard. Since `HybridChunker` uses its own `bert-base-uncased` tokenizer internally, and the qwen3-embedding context window is 8192 tokens, the custom tokenizer wrapper was removed entirely. Chunk size is now set to 512 tokens via `CHUNK_MAX_TOKENS`.

### Thinking mode

`qwen3.5:4b` has a built-in reasoning/thinking mode that adds ~3 minutes of latency per response. It is disabled via `extra_body={"think": False}` in every chat completion call.

### API key

The OpenAI SDK is used as the HTTP client for both chat and embeddings (pointed at `http://ollama:11434/v1`). Ollama requires a non-empty `api_key` field; the literal string `"ollama"` is passed as a dummy value.

## Alternatives Considered

| Model | Dims | Reason not chosen |
|---|---|---|
| `mxbai-embed-large` | 1024 | Not already on the server |
| `nomic-embed-text` | 768 | Not already on the server |
| `text-embedding-3-large` | 3072 | External API, costs money, sends data off-device |
