# Runbook operativo YT-KB

## Prerequisiti
- Docker + Docker Compose, oppure Postgres 16 con estensioni `vector` e `pg_trgm`.
- `.env` copiato da `.env.example`. Di default la pipeline è **CPU-only e offline**
  (embedding provider `fake`, chat disabilitata), quindi funziona senza chiavi cloud.

## Migrazioni
```bash
make migrate                 # alembic upgrade head
cd backend && alembic downgrade -1
cd backend && alembic revision --autogenerate -m "descrizione"
```
La dimensione embedding è **fissata a 1536 nella migrazione 0001**. Cambiarla è una
procedura esplicita: nuova colonna → backfill → swap → drop (mai in-place).

## Health check
```bash
curl -s localhost:8000/health | jq
# status=ok e extensions.vector=true, extensions.pg_trgm=true
```
`status=degraded` → un'estensione manca: esegui `db/init/01-extensions.sql` sul DB.

## Ingestion
```bash
ytkb add-channel @creator
ytkb ingest @creator --limit 20     # discovery video/shorts + trascrizioni
ytkb status
```
Idempotente: rerun non duplica canali (`yt_channel_id`), video (`yt_video_id`) o
unità (`content_hash`). In caso di rate limiting YouTube: configura
`YT_DLP_COOKIES_FILE` / `YT_DLP_PROXY` e affidati agli scan notturni.

## Ricerca da psql (senza LLM)
```sql
-- kNN semantica (cosine, indice HNSW)
SELECT id, text FROM chunks ORDER BY embedding <=> :query_vec LIMIT 10;
-- full-text italiano + inglese (indice GIN su tsv generata)
SELECT id, text FROM chunks WHERE tsv @@ websearch_to_tsquery('italian', :q);
-- fuzzy trigram
SELECT id FROM chunks WHERE similarity(text, :q) > 0.1 ORDER BY similarity(text, :q) DESC;
```

## Backup / restore
```bash
docker compose exec postgres pg_dump -U ytkb ytkb > backup.sql   # copre TUTTO
cat backup.sql | docker compose exec -T postgres psql -U ytkb ytkb
```

## Troubleshooting
- **`vector`/`pg_trgm` mancanti**: l'immagine `pgvector/pgvector:pg16` + init script
  li abilitano; la migrazione `0001` esegue comunque `CREATE EXTENSION IF NOT EXISTS`.
- **Chat non risponde**: verifica `CHAT_ENABLED=true` e `ANTHROPIC_API_KEY`. Con chat
  disabilitata il router `/chat` non viene montato: è un comportamento atteso.
- **Whisper CPU lento**: è un fallback batch notturno; limita con `ASR_MAX_DURATION_S`
  e dai priorità agli Shorts, oppure usa un provider ASR cloud.
