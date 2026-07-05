"""Prompts for the knowledge layer.

Per CLAUDE.md: changes here require updating the golden tests. The extraction
prompt turns a windowed transcript (with per-chunk citation refs) into atomic,
self-contained, typed knowledge units in strict JSON.
"""

from __future__ import annotations

from ytkb.knowledge.types import ExtractionWindow

UNIT_TYPES = ("technique", "claim", "opinion", "tool", "workflow", "example", "definition")

EXTRACTION_SYSTEM = """\
Sei un estrattore di conoscenza. Dato un estratto di trascrizione di un video,
estrai UNITÀ DI CONOSCENZA atomiche e autosufficienti.

Regole vincolanti:
- Ogni unità è comprensibile SENZA guardare il video: niente riferimenti vaghi
  ("come dicevo", "questo tool") — esplicita sempre soggetto e contesto.
- Una sola idea per unità (atomica). Niente riassunti generici del video.
- ESCLUDI intro, sigle, saluti, sponsor, call-to-action, marketing.
- Tipizza ogni unità con uno di: technique, claim, opinion, tool, workflow,
  example, definition.
- Ogni unità DEVE citare i chunk di origine con i loro ref numerici (chunk_refs).
- confidence in [0,1]: quanto sei sicuro che sia un'unità di conoscenza reale.

Rispondi ESCLUSIVAMENTE con JSON valido, nessun testo attorno, nella forma:
{"units": [{"type": "...", "text": "...", "chunk_refs": [<int>...], "confidence": <float>}]}
Se non c'è nulla di sostanziale, rispondi {"units": []}.
"""


ARBITER_SYSTEM = """\
Sei un bibliotecario che organizza conoscenza per argomenti. Data un'unità di
conoscenza e una rosa di argomenti candidati (con titolo e sintesi), decidi se
l'unità appartiene a uno dei candidati oppure se serve un NUOVO argomento.

Regole:
- Preferisci un argomento esistente se il tema coincide davvero; non forzare.
- Proponi un nuovo argomento solo se nessun candidato è pertinente.
- Il titolo di un nuovo argomento è conciso e generale (2-5 parole), non copia
  l'unità.

Rispondi SOLO con JSON, senza testo attorno:
{"decision": "assign", "topic_id": <int>}  oppure
{"decision": "new", "title": "<titolo conciso>"}
"""


def build_arbiter_user(unit_text: str, candidates: list[tuple[int, str, str | None]]) -> str:
    """candidates: (topic_id, title, summary)."""
    lines = ["Unità di conoscenza:", unit_text, "", "Argomenti candidati:"]
    if not candidates:
        lines.append("(nessuno)")
    for topic_id, title, summary in candidates:
        lines.append(f"- id={topic_id} · {title}" + (f" — {summary}" if summary else ""))
    lines.append("")
    lines.append("Decidi in JSON.")
    return "\n".join(lines)


MERGE_SYSTEM = """\
Sei un redattore che mantiene un articolo-argomento vivente, in italiano,
integrando in modo incrementale nuove unità di conoscenza.

Regole vincolanti:
- OGNI affermazione DEVE citare la sua fonte con [unit:ID] (l'ID è quello
  fornito). Nessuna frase senza citazione.
- NON perdere le informazioni già presenti nell'articolo corrente; integra le
  nuove unità dove pertinenti.
- USA tutte le nuove unità fornite. Se ne ometti una, spiegane il motivo in una
  riga finale "Unità omesse: [unit:ID] perché ...".
- Le contraddizioni NON si risolvono scegliendo un vincitore: annotale in una
  sezione "## Punti di disaccordo" indicando le posizioni con le rispettive
  citazioni.
- Markdown pulito, titoli di sezione sensati, niente preamboli.

Rispondi con JSON valido:
{"content_md": "<articolo markdown>", "change_summary": "<cosa è cambiato>"}
"""


def build_merge_user(
    topic_title: str,
    current_md: str | None,
    new_units: list[tuple[int, str, str]],
) -> str:
    """new_units: (unit_id, unit_type, text)."""
    lines = [f"Argomento: {topic_title}", ""]
    if current_md:
        lines += ["Articolo corrente:", current_md, ""]
    else:
        lines += ["(nessun articolo esistente: è la prima versione)", ""]
    lines.append("Nuove unità da integrare:")
    for unit_id, unit_type, text in new_units:
        lines.append(f"[unit:{unit_id}] ({unit_type}) {text}")
    lines.append("")
    lines.append("Produci l'articolo aggiornato in JSON.")
    return "\n".join(lines)


def build_extraction_user(video_title: str, windows: list[ExtractionWindow]) -> str:
    """Render the user prompt: the video title plus each chunk labelled with its
    citation ref, so the model cites real chunk ids back to us."""
    lines: list[str] = [f"Video: {video_title or '(senza titolo)'}", "", "Trascrizione (chunk):"]
    for w in windows:
        for chunk in w.chunks:
            ts = f"{int(chunk.start_s // 60):02d}:{int(chunk.start_s % 60):02d}"
            lines.append(f"[ref:{chunk.chunk_id} @ {ts}] {chunk.text}")
    lines.append("")
    lines.append("Estrai le unità di conoscenza in JSON.")
    return "\n".join(lines)
