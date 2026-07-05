# YT-KB — Regole di progetto

Memoria totale per argomento da contenuti YouTube. Non è un semplice RAG
"chunk di video → chat": è una **memoria organizzata per argomenti** che
decompone i video in unità di conoscenza atomiche e le fonde incrementalmente
in articoli-argomento viventi, versionati, con provenienza completa.

## Stack
- Python 3.12 (dev supporta 3.11), FastAPI, SQLAlchemy 2 async, Pydantic v2,
  Postgres 16 + pgvector. Frontend React + Vite + TS.
- UN SOLO datastore: Postgres. Vietato introdurre altri store (no Chroma, no
  Redis) senza discussione.

## Regole vincolanti
- Dimensione embedding fissata in migrazione `0001`: cambiarla richiede
  migrazione + reindex esplicito (procedura: nuova colonna → backfill → swap →
  drop). Default: **1536** (`text-embedding-3-small`).
- Ogni modifica schema via Alembic. Idempotenza obbligatoria in tutta la
  pipeline (ID deterministici, `content_hash`).
- Il layer conoscenza è append-only sulle versioni: mai UPDATE distruttivi su
  `topic_articles`.
- Ogni affermazione negli articoli DEVE citare `[unit:ID]`; il validatore in
  `knowledge/merge.py` è vincolante.
- Gli endpoint `/kb/*` non devono MAI chiamare un LLM. La chat è isolata in
  `rag/` e disattivabile via env (`CHAT_ENABLED=false`).
- Prompt in `knowledge/prompts.py` e `rag/prompts.py`: modifiche solo con
  aggiornamento dei golden test.
- CPU-only: nessuna dipendenza CUDA/ROCm. `faster-whisper` int8, embeddings via
  API di default.
- `make dev` / `make test` / `make lint` devono passare prima di chiudere un
  task.

## Architettura a due layer
```
LAYER FONTE (provenienza, immutabile)
  channels → videos → transcripts → chunks (+ embedding pgvector)
           │  estrazione LLM (batch, Haiku)
           ▼
LAYER CONOSCENZA (vivente, incrementale)
  knowledge_units (+ embedding)
           │  assegnazione argomento (similarità + arbitraggio LLM)
           ▼
  topics ──── topic_articles (versionati, markdown, con citazioni)
```

## Layout
- `backend/src/ytkb/config.py` — settings Pydantic.
- `backend/src/ytkb/db/` — modelli, repository, ricerca ibrida SQL.
- `backend/src/ytkb/ingestion/` — providers, transcripts, scheduler.
- `backend/src/ytkb/indexing/` — chunker, embeddings.
- `backend/src/ytkb/knowledge/` — extraction, assignment, merge, taxonomy.
- `backend/src/ytkb/rag/` — chat (opzionale).
- `backend/src/ytkb/api/` — routers: kb (no-LLM), admin, chat.
- `backend/src/ytkb/cli.py` — comandi operativi.
