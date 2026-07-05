import { useEffect, useState } from "react";
import { getHealth, listTopics, type Health, type Topic } from "./api";

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(String(e)));
    listTopics().then(setTopics).catch(() => void 0);
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 820, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>YT-KB</h1>
      <p style={{ color: "#666" }}>Memoria totale per argomento da contenuti YouTube</p>

      <section>
        <h2>Stato backend</h2>
        {error && <p style={{ color: "crimson" }}>Errore: {error}</p>}
        {health ? (
          <ul>
            <li>Stato: <strong>{health.status}</strong></li>
            <li>DB: {health.db}</li>
            <li>Embedding dim: {health.embedding_dim}</li>
            <li>Chat abilitata: {String(health.chat_enabled)}</li>
            <li>
              Estensioni:{" "}
              {Object.entries(health.extensions)
                .map(([k, v]) => `${k}=${v ? "✓" : "✗"}`)
                .join(", ")}
            </li>
          </ul>
        ) : (
          !error && <p>Caricamento…</p>
        )}
      </section>

      <section>
        <h2>Argomenti ({topics.length})</h2>
        {topics.length === 0 ? (
          <p style={{ color: "#888" }}>
            Nessun argomento ancora. Ingerisci dei canali per popolare la knowledge base.
          </p>
        ) : (
          <ul>
            {topics.map((t) => (
              <li key={t.id}>
                <strong>{t.title}</strong> <em>({t.status})</em> — {t.units_count} unità
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
