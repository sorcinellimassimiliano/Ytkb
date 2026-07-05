"""Chat prompts. Per CLAUDE.md, changes require updating the golden tests.

The chat layer is grounded strictly in the retrieved context and must cite its
sources as [titolo — mm:ss]. It is honest when the context is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Source:
    label: str  # "titolo — mm:ss"
    kind: str  # topic | unit | chunk
    ref: str  # slug or id
    url: str | None = None


CHAT_SYSTEM = """\
Sei l'assistente di una knowledge base costruita da video YouTube. Rispondi in
italiano, in modo conciso e concreto.

Regole vincolanti:
- Usa ESCLUSIVAMENTE le informazioni nel CONTESTO fornito. Non inventare.
- Cita le fonti tra parentesi quadre nel formato [titolo — mm:ss] quando presenti
  nel contesto.
- Se il contesto non basta a rispondere, dillo apertamente invece di indovinare.
- Preferisci le sintesi degli argomenti per il quadro d'insieme e le unità/chunk
  per i dettagli puntuali.
"""


def build_chat_prompt(query: str, context: str) -> str:
    return (
        f"CONTESTO:\n{context}\n\n"
        f"DOMANDA: {query}\n\n"
        "Rispondi usando solo il contesto e citando le fonti."
    )
