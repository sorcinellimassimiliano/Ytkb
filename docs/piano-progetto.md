# Piano di Progetto v2 — YT-KB: Memoria totale per argomento da contenuti YouTube

> Documento di specifica destinato a Claude Code. Revisione v2 con: **pgvector** come unico store (niente ChromaDB), pipeline **CPU-only** (nessuna GPU), **KB navigabile senza LLM**, e soprattutto **aggregazione incrementale per argomento** invece che per video.

---

## 1. Obiettivo e cambio di paradigma

Il sistema non è un semplice RAG "chunk di video → chat". È una **memoria totale organizzata per argomenti**:

- Ogni video ingerito viene **decomposto in unità di conoscenza atomiche** (tecniche, claim, opinioni, tool, pattern, workflow).
- Le unità vengono **assegnate ad argomenti** (esistenti o nuovi) e **fuse incrementalmente** in articoli-argomento viventi, versionati, con provenienza completa (video + timestamp per ogni affermazione).
- Il video è solo la *fonte*; l'unità di consultazione è l'**argomento**.
- La KB è **navigabile e ricercabile senza alcun LLM** (albero argomenti, articoli, full-text search, ricerca semantica). La chat LLM è un layer opzionale sopra.

**Esempio concreto:** 5 creator parlano di "context management negli agenti" in 12 video diversi nell'arco di mesi. Il sistema mantiene UN articolo "Context management" che cresce a ogni nuovo video, integra le tecniche nuove, segnala le opinioni contrastanti con data e fonte, e linka ogni paragrafo al momento esatto del video di origine.

**Non-obiettivi v1:** Instagram Reels (interfaccia `SourceProvider` astratta predisposta), multi-tenancy, download video completi.

---

## 2. Stack tecnologico (rivisto)

| Layer | Scelta | Note |
|---|---|---|
| DB unico | **PostgreSQL 16 + pgvector 0.7+** | Vettori, FTS, relazionale, stato pipeline: tutto in un posto. Indici HNSW |
| Ingestion | Python 3.12, `yt-dlp`, `youtube-transcript-api` | Come v1 |
| Fallback ASR (CPU) | `faster-whisper` modello `small`/`medium` int8 su CPU, **oppure** ASR cloud (AssemblyAI / OpenAI) configurabile | Nessuna GPU: whisper CPU è lento ma è un job batch notturno; per canali prioritari valutare cloud |
| Embeddings | **API cloud di default** (`text-embedding-3-small` o Voyage) — CPU-only rende i modelli locali grossi poco pratici; opzione locale `bge-small` / `multilingual-e5-small` (dim 384) per chi vuole zero cloud | Dimensione vettore fissata a livello migrazione: scegliere PRIMA (1536 o 384) |
| LLM sintesi/chat | Claude API (Haiku per estrazione unità, Sonnet per merge articoli e chat) — https://docs.claude.com/en/api/overview | Astratto dietro `LLMClient` |
| Backend | FastAPI + Pydantic v2 + SQLAlchemy 2 async + Alembic | |
| Scheduler | APScheduler in-process | |
| Frontend | React + Vite + TS | KB browser + chat |
| Deploy | Docker Compose (postgres, backend, frontend, worker) | Whisper worker = container CPU con limite risorse |

---

## 3. Architettura a due layer

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
     │
     └── gerarchia (parent_id) + relazioni (related_topics)
```

### Pipeline incrementale per video (macchina a stati)

```
pending → transcribed → chunked → embedded → units_extracted → assigned → merged
                                                   │ (per ogni unità)
                                                   ▼
                          match embedding vs centroidi topic (top-3)
                          score > θ_high  → assegna al topic
                          θ_low..θ_high   → arbitraggio LLM (assegna o proponi nuovo)
                          score < θ_low   → proponi nuovo topic
                                                   ▼
                          job di merge per topic "sporco":
                          articolo_corrente + nuove unità → LLM → nuova versione articolo
                          (diff salvato, contraddizioni annotate con fonte e data)
```

Proprietà chiave:
- **Idempotenza**: ID deterministici ovunque (`yt_video_id:start_s` per chunk, hash contenuto per unità); rerun sicuri.
- **Incrementalità**: il merge riceve solo le unità nuove, non riprocessa mai tutto l'articolo da zero. Re-sintesi completa disponibile come comando esplicito (`rebuild-topic`).
- **Provenienza obbligatoria**: ogni unità porta `chunk_ids` → l'articolo cita `[unit:ID]`, il renderer risolve in "titolo video — mm:ss" cliccabile. Il prompt di merge vieta affermazioni senza citazione.
- **Contraddizioni**: il merge non sceglie un vincitore; annota "X sostiene A (video, data), Y sostiene B (video, data)".
- **Deriva tassonomia**: comando periodico `topics-review` che rileva topic troppo simili (distanza centroidi) e propone fusioni; le fusioni sono operazioni esplicite approvate da admin UI, mai automatiche.

### Modello dati (Postgres, tutto con Alembic)

```sql
channels(id, yt_channel_id, handle, title, active, last_checked_at, ...)
videos(id, yt_video_id UNIQUE, channel_id, title, published_at, duration_s,
       kind ENUM('video','short'), url, ingest_status, error_msg, ...)
transcripts(id, video_id, language, source ENUM('yt_manual','yt_auto','whisper','cloud_asr'),
            raw_json JSONB, ...)
chunks(id, video_id, start_s, end_s, text, token_count,
       embedding vector(1536),          -- HNSW index, cosine
       tsv tsvector GENERATED,          -- GIN index (FTS)
       ...)
knowledge_units(id, content_hash UNIQUE, video_id, chunk_ids INT[], 
       unit_type ENUM('technique','claim','opinion','tool','workflow','example','definition'),
       text, embedding vector(1536), topic_id NULL, assignment ENUM('auto','llm','manual','pending'),
       confidence, created_at)
topics(id, slug UNIQUE, title, summary, parent_id NULL,
       centroid vector(1536),           -- media incrementale degli embedding delle unità
       status ENUM('active','proposed','merged_into'), merged_into_id NULL,
       units_count, last_merged_at, ...)
topic_articles(id, topic_id, version, content_md, units_included INT[],
       change_summary, created_at)      -- append-only, l'ultima versione è la corrente
topic_relations(topic_a, topic_b, kind)
chat_sessions / chat_messages(..., sources JSONB)
```

Ricerca senza LLM, tutta in Postgres:
- **FTS** (tsvector + GIN, dizionari `italian`+`english`+`simple`, pg_trgm per fuzzy) su articoli, unità, trascrizioni.
- **Semantica**: embed della query (una chiamata embedding, nessun LLM) → kNN pgvector su unità e chunk.
- **Ibrida**: RRF (reciprocal rank fusion) FTS + vettoriale, implementata in SQL/servizio.

---

## 4. Fasi di sviluppo

### Fase 0 — Bootstrap (0,5 g)
Monorepo, Docker Compose (postgres con estensioni `vector`+`pg_trgm` via init script, backend, frontend), Alembic con migrazione iniziale completa (decidere qui la dimensione embedding!), Makefile, CI Gitea (ruff, mypy, pytest).
**Accettazione:** `docker compose up` OK, `/health` verifica estensioni attive.

### Fase 1 — Ingestion fonte (2–3 g)
Come v1: `YouTubeProvider` (video + Shorts via yt-dlp flat-playlist), trascrizioni via `youtube-transcript-api` (manuale > auto, `['it','en']`), backoff + jitter, stati, CLI (`ingest`, `status`), scheduler.
**Accettazione:** canale reale → video in `transcribed`, rerun idempotente.

### Fase 2 — Chunking + embedding su pgvector (1–2 g)
Chunker timestamp-aware (400–600 token, overlap 15%, taglio su frasi). Embedding batched con retry → colonna `chunks.embedding`, indice HNSW. FTS generata. Comando `reindex-video`.
**Accettazione:** kNN e FTS su chunk funzionanti da psql; test proprietà chunker.

### Fase 3 — Estrazione unità di conoscenza (2–3 g) ★ nuovo
- Prompt di estrazione (Haiku, batch API per costi): input = trascrizione a finestre con timestamp; output = JSON di unità `{type, text, chunk_refs, confidence}`. Regole: unità atomiche e autosufficienti (comprensibili senza il video), no riassunti generici, no marketing/sponsor/intro.
- Dedup per `content_hash` + near-dup semantico (soglia cosine) prima dell'insert.
- Embedding delle unità.
- Golden test: 2–3 trascrizioni fixture con unità attese, verifica strutturale (non exact-match sul testo).
**Accettazione:** da un video reale escono unità sensate, tipizzate, con riferimenti chunk validi; rerun non duplica.

### Fase 4 — Assegnazione argomenti + merge incrementale (3–4 g) ★ cuore del sistema
- Matcher: query kNN unità→centroidi topic; soglie `θ_high`/`θ_low` in config; arbitraggio LLM nella fascia intermedia; proposta nuovi topic (stato `proposed`, promozione automatica a `active` oltre N unità o via admin).
- Aggiornamento centroide incrementale a ogni assegnazione.
- Merge worker: per ogni topic con unità non ancora incluse → prompt di merge (Sonnet): articolo corrente + nuove unità → nuova versione markdown con citazioni `[unit:ID]` obbligatorie, sezione "Punti di disaccordo" per contraddizioni, `change_summary`. Validatore post-merge: ogni `[unit:ID]` citato esiste, nessuna unità nuova ignorata (o motivazione).
- Comandi: `rebuild-topic <slug>` (re-sintesi da zero), `topics-review` (proposte fusione), merge topic manuale con remap unità.
**Accettazione:** ingerendo 3 video sullo stesso tema in momenti diversi, l'articolo cresce a ogni ingestione mantenendo le versioni; le citazioni risolvono a timestamp corretti; nessuna unità orfana.

### Fase 5 — KB browser senza LLM (2–3 g) ★ nuovo
Frontend + endpoint REST puri (zero LLM):
- **Albero argomenti** (gerarchia + correlati), pagina argomento = articolo corrente renderizzato, citazioni cliccabili → YouTube al timestamp, cronologia versioni con diff.
- **Ricerca**: barra unica con tre modalità — full-text, semantica, ibrida (RRF) — risultati raggruppati per tipo (argomenti / unità / chunk trascrizione).
- **Indice fonti**: canali → video → lettore trascrizione con timestamp e link alle unità estratte da quel video ("dove è finito questo contenuto").
- **Admin**: watchlist canali, stato pipeline per video, topic proposti da approvare, fusioni topic.
**Accettazione:** con backend LLM spento (env senza chiave chat), navigazione e ricerca completamente funzionanti.

### Fase 6 — Chat LLM opzionale (1–2 g)
`POST /chat` SSE. Retrieval privilegiato: prima articoli-argomento pertinenti (già sintetizzati = contesto denso), poi unità/chunk per dettagli. Prompt: italiano, citazioni `[titolo — mm:ss]`, onestà su contesto insufficiente. Sessioni persistite.
**Accettazione:** risposta con citazioni valide; feature disattivabile via config senza rompere il resto.

### Fase 7 — Fallback ASR CPU/cloud (1–2 g)
Worker per video `no_transcript`: `yt-dlp -x` solo audio → faster-whisper `small`/`medium` int8 su CPU (limite durata configurabile, priorità Shorts che sono brevi) **oppure** provider cloud (AssemblyAI/OpenAI) selezionabile via env. Rientro automatico in Fase 2.
**Accettazione:** video senza sottotitoli diventa interrogabile end-to-end.

### Fase 8 — Hardening (1–2 g)
- Valutazione: 15–20 domande gold → il topic/unità atteso è nel top-k (baseline prima di ottimizzare soglie).
- Costi: report LLM/embedding per ingestione (token contati), batch API per estrazione.
- Osservabilità: structlog, `/admin/stats` (video/unità/topic, failure rate, coda merge).
- Backup: pg_dump copre TUTTO (unico vantaggio enorme del consolidamento su Postgres).
- Runbook in `docs/`.

---

## 5. Struttura repository

```
ytkb/
├── CLAUDE.md
├── docker-compose.yml          # postgres(+pgvector), backend, frontend, asr-worker
├── Makefile
├── backend/src/ytkb/
│   ├── config.py
│   ├── db/                     # modelli, repository, ricerca ibrida SQL
│   ├── ingestion/              # providers, transcripts, scheduler
│   ├── indexing/               # chunker, embeddings
│   ├── knowledge/              # ★ extraction, assignment, merge, taxonomy
│   ├── rag/                    # chat (opzionale)
│   ├── api/                    # routers: kb (no-LLM), admin, chat
│   └── cli.py
├── asr-worker/
├── frontend/src/
└── docs/
```

---

## 6. CLAUDE.md suggerito

```markdown
# YT-KB — Regole di progetto
- Python 3.12, FastAPI, SQLAlchemy 2 async, Pydantic v2, Postgres 16 + pgvector. Frontend React+Vite+TS.
- UN SOLO datastore: Postgres. Vietato introdurre altri store (no Chroma, no Redis) senza discussione.
- Dimensione embedding fissata in migrazione 0001: cambiarla richiede migrazione + reindex esplicito.
- Ogni modifica schema via Alembic. Idempotenza obbligatoria in tutta la pipeline (ID deterministici, content_hash).
- Il layer conoscenza è append-only sulle versioni: mai UPDATE distruttivi su topic_articles.
- Ogni affermazione negli articoli DEVE citare [unit:ID]; il validatore in knowledge/merge.py è vincolante.
- Gli endpoint /kb/* non devono MAI chiamare un LLM. La chat è isolata in rag/ e disattivabile via env.
- Prompt in knowledge/prompts.py e rag/prompts.py: modifiche solo con aggiornamento dei golden test.
- CPU-only: nessuna dipendenza CUDA/ROCm. faster-whisper int8, embeddings via API di default.
- make dev / make test / make lint devono passare prima di chiudere un task.
```

---

## 7. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Rate limiting YouTube su trascrizioni da IP datacenter | Backoff aggressivo, scan notturni, cookie/proxy configurabili in yt-dlp; è il punto di rottura più frequente di questi sistemi |
| **Deriva della tassonomia** (topic duplicati, frammentazione) | Soglie configurabili + arbitraggio LLM, `topics-review` periodico, fusioni solo manuali da admin, centroidi aggiornati incrementalmente |
| **Degradazione articoli dopo molti merge** (perdita struttura, verbosità) | Validatore post-merge, `change_summary` obbligatorio, `rebuild-topic` per re-sintesi pulita da tutte le unità, versioni sempre reversibili |
| Costi LLM in ingestion (estrazione+merge su ogni video) | Haiku + batch API per estrazione; merge solo su topic "sporchi"; report costi per ingestione in admin |
| Whisper CPU lento | Solo fallback, batch notturno, limite durata, priorità Shorts; opzione ASR cloud per canali importanti |
| Cambio modello embedding | Collection versionata a livello colonna? No: dim fissa; procedura documentata: nuova colonna → backfill → swap → drop |

---

## 8. Stima

| Fase | Effort |
|---|---|
| 0 Bootstrap | 0,5 g |
| 1 Ingestion | 2–3 g |
| 2 Chunk+pgvector | 1–2 g |
| 3 Estrazione unità ★ | 2–3 g |
| 4 Topic + merge ★ | 3–4 g |
| 5 KB browser ★ | 2–3 g |
| 6 Chat | 1–2 g |
| 7 ASR fallback | 1–2 g |
| 8 Hardening | 1–2 g |
| **Totale** | **~14–20 giorni/uomo** |

---

## 9. Ordine di lavoro per Claude Code

1. Fase 0 + migrazione completa dello schema (tutto lo schema subito: evita migrazioni a catena).
2. Fase 1, test su canale reale.
3. Fase 2 (chunker con test di proprietà).
4. Fase 3 su fixture prima, poi su dati reali; congelare i golden test.
5. Fase 4 in due sessioni: (a) matcher+assegnazione, (b) merge+validatore. È la parte a maggior rischio: iterare sui prompt con i golden test prima di scalare.
6. Fase 5 (prima le API /kb, poi il frontend).
7. Fasi 6–8.

Regola: ogni sessione si chiude con `make test && make lint` verdi e commit atomico. La Fase 4 non si considera chiusa finché il test end-to-end "3 video → 1 articolo che cresce" non passa.
