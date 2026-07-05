import { useEffect, useState } from "react";
import {
  chatStream,
  getHealth,
  getSources,
  getTopic,
  getTree,
  getVersion,
  getVersions,
  getVideo,
  search,
  youtubeUrl,
  formatTs,
  type Health,
  type Source,
  type TopicDetail,
  type TreeNode,
  type VersionInfo,
  type VideoPage,
  type SearchResults,
  type Article,
} from "./api";
import { Markdown } from "./Markdown";

// --- tiny hash router ------------------------------------------------------
type Route =
  | { view: "home" }
  | { view: "topic"; slug: string }
  | { view: "search"; q: string }
  | { view: "sources" }
  | { view: "video"; ytId: string }
  | { view: "chat" };

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  const [head, ...rest] = h.split("/");
  const tail = rest.join("/");
  if (head === "topic" && tail) return { view: "topic", slug: decodeURIComponent(tail) };
  if (head === "search" && tail) return { view: "search", q: decodeURIComponent(tail) };
  if (head === "video" && tail) return { view: "video", ytId: decodeURIComponent(tail) };
  if (head === "sources") return { view: "sources" };
  if (head === "chat") return { view: "chat" };
  return { view: "home" };
}

function useRoute(): Route {
  const [route, setRoute] = useState<Route>(parseHash());
  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

const go = (path: string) => {
  window.location.hash = path;
};

// --- app -------------------------------------------------------------------
export function App() {
  const route = useRoute();
  const [q, setQ] = useState("");
  const { data: health } = useAsync<Health>(getHealth, []);

  return (
    <div className="app">
      <header>
        <a className="brand" href="#/" onClick={() => go("/")}>
          YT-KB
        </a>
        <nav>
          <a href="#/">Argomenti</a>
          <a href="#/sources">Fonti</a>
          {health?.chat_enabled && <a href="#/chat">Chat</a>}
        </nav>
        <form
          className="searchbar"
          onSubmit={(e) => {
            e.preventDefault();
            if (q.trim()) go(`/search/${encodeURIComponent(q.trim())}`);
          }}
        >
          <input
            placeholder="Cerca argomenti, unità, trascrizioni…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <button type="submit">Cerca</button>
        </form>
      </header>
      <main>
        {route.view === "home" && <HomeView />}
        {route.view === "topic" && <TopicView slug={route.slug} />}
        {route.view === "search" && <SearchView q={route.q} />}
        {route.view === "sources" && <SourcesView />}
        {route.view === "video" && <VideoView ytId={route.ytId} />}
        {route.view === "chat" && <ChatView />}
      </main>
      <Footer health={health} />
    </div>
  );
}

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  sources?: { label: string; kind: string; ref: string }[];
}

function ChatView() {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState<number | null>(null);

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", content: text }, { role: "assistant", content: "" }]);
    try {
      await chatStream(text, sessionId, (e) => {
        if (e.type === "session" && e.session_id != null) setSessionId(e.session_id);
        else if (e.type === "token" && e.text) {
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              content: copy[copy.length - 1].content + e.text,
            };
            return copy;
          });
        } else if (e.type === "done") {
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = { ...copy[copy.length - 1], sources: e.sources };
            return copy;
          });
        }
      });
    } catch (err) {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: `Errore: ${String(err)}` };
        return copy;
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section>
      <h1>Chat</h1>
      <p className="muted">Risposte basate solo sulla knowledge base, con citazioni.</p>
      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            <div className="msg-body">{m.content || (busy && i === messages.length - 1 ? "…" : "")}</div>
            {m.sources && m.sources.length > 0 && (
              <div className="msg-sources muted">
                Fonti: {m.sources.map((s) => s.label).join(" · ")}
              </div>
            )}
          </div>
        ))}
      </div>
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <input
          placeholder="Fai una domanda…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy}>
          Invia
        </button>
      </form>
    </section>
  );
}

function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    fn()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error };
}

function TreeItem({ node }: { node: TreeNode }) {
  return (
    <li>
      <a href={`#/topic/${encodeURIComponent(node.slug)}`}>{node.title}</a>{" "}
      <span className="muted">
        · {node.units_count} unità · {node.status}
      </span>
      {node.children.length > 0 && (
        <ul>
          {node.children.map((c) => (
            <TreeItem key={c.id} node={c} />
          ))}
        </ul>
      )}
    </li>
  );
}

function HomeView() {
  const { data: tree, error } = useAsync<TreeNode[]>(getTree, []);
  return (
    <section>
      <h1>Argomenti</h1>
      {error && <p className="error">Errore: {error}</p>}
      {tree && tree.length === 0 && (
        <p className="muted">
          Nessun argomento. Esegui la pipeline (ingest → index → extract-units → assign → merge).
        </p>
      )}
      {tree && tree.length > 0 && (
        <ul className="tree">
          {tree.map((n) => (
            <TreeItem key={n.id} node={n} />
          ))}
        </ul>
      )}
    </section>
  );
}

function TopicView({ slug }: { slug: string }) {
  const { data: topic, error } = useAsync<TopicDetail>(() => getTopic(slug), [slug]);
  const { data: versions } = useAsync<VersionInfo[]>(() => getVersions(slug), [slug]);
  const [compareTo, setCompareTo] = useState<number | null>(null);
  const { data: older } = useAsync<Article | null>(
    () => (compareTo == null ? Promise.resolve(null) : getVersion(slug, compareTo)),
    [slug, compareTo],
  );

  if (error) return <p className="error">Errore: {error}</p>;
  if (!topic) return <p>Caricamento…</p>;

  return (
    <section>
      <h1>{topic.title}</h1>
      <p className="muted">
        {topic.status} · {topic.units_count} unità
        {topic.article ? ` · versione ${topic.article.version}` : ""}
      </p>

      {topic.related.length > 0 && (
        <p>
          Correlati:{" "}
          {topic.related.map((r, i) => (
            <span key={r.id}>
              {i > 0 && ", "}
              <a href={`#/topic/${encodeURIComponent(r.slug)}`}>{r.title}</a>
            </span>
          ))}
        </p>
      )}

      {versions && versions.length > 1 && (
        <div className="versionbar">
          <label>
            Confronta con versione:{" "}
            <select
              value={compareTo ?? ""}
              onChange={(e) => setCompareTo(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">—</option>
              {versions
                .filter((v) => !topic.article || v.version !== topic.article.version)
                .map((v) => (
                  <option key={v.version} value={v.version}>
                    v{v.version} · {v.change_summary}
                  </option>
                ))}
            </select>
          </label>
        </div>
      )}

      {!topic.article && <p className="muted">Nessun articolo ancora sintetizzato per questo argomento.</p>}

      {topic.article && compareTo == null && <Markdown text={topic.article.content_md} />}
      {topic.article && older && (
        <Diff oldText={older.content_md} newText={topic.article.content_md} oldV={older.version} newV={topic.article.version} />
      )}

      {versions && versions.length > 0 && (
        <>
          <h3>Cronologia versioni</h3>
          <ul className="muted">
            {versions.map((v) => (
              <li key={v.version}>
                v{v.version} · {v.units_included} unità · {v.change_summary}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function Diff({ oldText, newText, oldV, newV }: { oldText: string; newText: string; oldV: number; newV: number }) {
  const oldLines = new Set(oldText.split("\n"));
  const newLines = new Set(newText.split("\n"));
  const added = newText.split("\n").filter((l) => l.trim() && !oldLines.has(l));
  const removed = oldText.split("\n").filter((l) => l.trim() && !newLines.has(l));
  return (
    <div className="diff">
      <p className="muted">
        Diff v{oldV} → v{newV}
      </p>
      {added.map((l, i) => (
        <div key={`a${i}`} className="diff-add">
          + {l}
        </div>
      ))}
      {removed.map((l, i) => (
        <div key={`r${i}`} className="diff-del">
          − {l}
        </div>
      ))}
      {added.length === 0 && removed.length === 0 && <p className="muted">Nessuna differenza testuale.</p>}
    </div>
  );
}

function SearchView({ q }: { q: string }) {
  const { data, error } = useAsync<SearchResults>(() => search(q), [q]);
  if (error) return <p className="error">Errore: {error}</p>;
  if (!data) return <p>Ricerca…</p>;
  const { topics, units, chunks } = data.results;
  return (
    <section>
      <h1>Risultati per «{q}»</h1>

      <h3>Argomenti ({topics.length})</h3>
      <ul>
        {topics.map((t) => (
          <li key={t.id}>
            <a href={`#/topic/${encodeURIComponent(t.slug)}`}>{t.title}</a>
          </li>
        ))}
      </ul>

      <h3>Unità di conoscenza ({units.length})</h3>
      <ul>
        {units.map((u) => (
          <li key={u.id}>
            <span className="badge">{u.unit_type}</span> {u.text}
          </li>
        ))}
      </ul>

      <h3>Trascrizioni ({chunks.length})</h3>
      <ul>
        {chunks.map((c) => (
          <li key={c.id}>
            <span className="muted">{formatTs(c.start_s)}</span> {c.text}
          </li>
        ))}
      </ul>

      {topics.length + units.length + chunks.length === 0 && <p className="muted">Nessun risultato.</p>}
    </section>
  );
}

function SourcesView() {
  const { data, error } = useAsync<Source[]>(getSources, []);
  if (error) return <p className="error">Errore: {error}</p>;
  if (!data) return <p>Caricamento…</p>;
  return (
    <section>
      <h1>Fonti</h1>
      {data.length === 0 && <p className="muted">Nessun canale registrato.</p>}
      <ul>
        {data.map((c) => (
          <li key={c.id}>
            <strong>{c.title ?? c.handle ?? c.yt_channel_id}</strong>{" "}
            <span className="muted">· {c.videos_total} video</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function VideoView({ ytId }: { ytId: string }) {
  const { data, error } = useAsync<VideoPage>(() => getVideo(ytId), [ytId]);
  if (error) return <p className="error">Errore: {error}</p>;
  if (!data) return <p>Caricamento…</p>;
  return (
    <section>
      <h1>{data.title ?? data.yt_video_id}</h1>
      <p className="muted">
        stato: {data.ingest_status} ·{" "}
        <a href={data.url ?? youtubeUrl(data.yt_video_id)} target="_blank" rel="noreferrer">
          apri su YouTube
        </a>
      </p>

      <h3>Unità estratte da questo video ({data.units.length})</h3>
      <ul>
        {data.units.map((u) => (
          <li key={u.id}>
            <span className="badge">{u.unit_type}</span> {u.text}
            {u.topic_slug && (
              <>
                {" "}
                → <a href={`#/topic/${encodeURIComponent(u.topic_slug)}`}>argomento</a>
              </>
            )}
          </li>
        ))}
      </ul>

      <h3>Trascrizione</h3>
      <div className="transcript">
        {data.segments.map((s, i) => (
          <p key={i}>
            <a href={youtubeUrl(data.yt_video_id, s.start_s)} target="_blank" rel="noreferrer" className="ts">
              {formatTs(s.start_s)}
            </a>{" "}
            {s.text}
          </p>
        ))}
      </div>
    </section>
  );
}

function Footer({ health }: { health: Health | null }) {
  if (!health) return null;
  return (
    <footer>
      YT-KB v{health.version} · DB {health.db} · embedding {health.embedding_dim} · chat{" "}
      {health.chat_enabled ? "on" : "off"}
    </footer>
  );
}
