# YT-KB — Memoria totale per argomento da contenuti YouTube

YT-KB non è un semplice RAG "chunk di video → chat". È una **memoria totale
organizzata per argomenti**: ogni video viene decomposto in unità di conoscenza
atomiche, assegnate ad argomenti e fuse incrementalmente in **articoli-argomento
viventi**, versionati, con provenienza completa (video + timestamp per ogni
affermazione). La KB è **navigabile e ricercabile senza alcun LLM**; la chat è un
layer opzionale.

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

Un **solo datastore**: PostgreSQL 16 + pgvector (vettori HNSW, FTS `tsvector`,
trigram, relazionale, stato pipeline). Pipeline **CPU-only** (embeddings via API
o modelli locali piccoli; ASR fallback via `faster-whisper` int8 o cloud).

## Stato di sviluppo

| Fase | Descrizione | Stato |
|---|---|---|
| 0 | Bootstrap: monorepo, Docker Compose, schema completo (Alembic), `/health`, CI | ✅ completata |
| 1 | Ingestion fonte (YouTube provider, trascrizioni, service idempotente, scheduler, CLI) | ✅ completata |
| 2 | Chunking timestamp-aware + embedding su pgvector + ricerca kNN/FTS/ibrida | ✅ completata |
| 3 | Estrazione unità di conoscenza (LLM astratto + euristico offline, golden test) | ✅ completata |
| 4a | Assegnazione argomenti (matcher kNN, soglie, arbitraggio, centroidi incrementali) | ✅ completata |
| 4b | Merge incrementale articoli versionati + validatore citazioni + topics-review | ✅ completata |
| 5 | KB browser senza LLM (API articoli/versioni/provenienza + frontend) | 🟡 API pronte, frontend base |
| 6 | Chat LLM opzionale | ⬜ |
| 7 | Fallback ASR CPU/cloud | ⬜ |
| 8 | Hardening | ⬜ |

Vedi [`docs/piano-progetto.md`](docs/piano-progetto.md) per il piano completo e
[`docs/runbook.md`](docs/runbook.md) per le operazioni.

## Avvio rapido (Docker)

```bash
cp .env.example .env
make dev          # postgres(+pgvector) + backend (migra e serve) + worker + frontend
```

- Backend: http://localhost:8000 — `GET /health` verifica DB ed estensioni.
- Frontend: http://localhost:5173
- Docs API: http://localhost:8000/docs

## Sviluppo backend (senza Docker)

Richiede un Postgres 16 con `vector` e `pg_trgm` (vedi `db/init/`).

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
export DATABASE_URL="postgresql+asyncpg://ytkb:ytkb@localhost:5432/ytkb"
alembic upgrade head
uvicorn ytkb.api.app:app --reload
```

## CLI

```bash
ytkb add-channel @creator             # registra un canale
ytkb ingest @creator --limit 20       # scopre video + scarica trascrizioni
ytkb ingest @creator --retry-errors   # riprova i video andati in errore
ytkb scan                             # una scansione ora per tutti i canali attivi
ytkb index --limit 50                 # chunk + embedding dei video trascritti
ytkb reindex-video <yt_video_id>      # re-chunk + re-embed pulito di un video
ytkb extract-units --limit 50         # estrae unità di conoscenza (embedded → units_extracted)
ytkb assign --limit 200               # assegna le unità agli argomenti (matcher + arbitraggio)
ytkb merge                            # fonde le nuove unità negli articoli-argomento (nuova versione)
ytkb rebuild-topic <slug>             # re-sintesi pulita dell'articolo da tutte le unità
ytkb topics-review                    # propone fusioni di argomenti quasi-duplicati (solo detection)
ytkb merge-topics <from> <into>       # fusione manuale di argomenti con remap unità
ytkb search "prompt caching" --mode hybrid   # ricerca chunk (fts|semantic|hybrid)
ytkb status                           # conteggi pipeline per stato
ytkb scheduler                        # scan notturni in-process (APScheduler)
```

> **Nota rete**: l'ingestione live richiede accesso a `youtube.com`. In ambienti
> con IP datacenter YouTube applica rate limiting: configura `YT_DLP_COOKIES_FILE`
> / `YT_DLP_PROXY`. La macchina a stati e l'idempotenza della pipeline sono
> coperte da test di integrazione contro Postgres reale (provider fittizio).

## Qualità

```bash
make test   # pytest
make lint   # ruff check + ruff format --check + mypy
```

Regole di progetto in [`CLAUDE.md`](CLAUDE.md). Vincoli chiave: un solo datastore,
dimensione embedding fissata in migrazione `0001`, layer conoscenza append-only,
`/kb/*` mai chiama un LLM, idempotenza ovunque.
