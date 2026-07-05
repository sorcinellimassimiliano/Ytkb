const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export interface Health {
  status: string;
  version: string;
  chat_enabled: boolean;
  embedding_dim: number;
  db: string;
  extensions: Record<string, boolean>;
}

export interface TreeNode {
  id: number;
  slug: string;
  title: string;
  status: string;
  units_count: number;
  children: TreeNode[];
}

export interface Article {
  version: number;
  content_md: string;
  change_summary: string | null;
  units_included: number[];
  created_at: string;
}

export interface TopicDetail {
  id: number;
  slug: string;
  title: string;
  status: string;
  units_count: number;
  related: { id: number; slug: string; title: string }[];
  article: Article | null;
}

export interface VersionInfo {
  version: number;
  change_summary: string | null;
  units_included: number;
  created_at: string;
}

export interface UnitProvenance {
  id: number;
  text: string;
  unit_type: string;
  topic_id: number | null;
  video: { yt_video_id: string; title: string | null; url: string | null } | null;
  chunks: { id: number; start_s: number; end_s: number }[];
}

export interface SearchResults {
  query: string;
  results: {
    topics: { id: number; slug: string; title: string; score: number }[];
    units: { id: number; text: string; unit_type: string; topic_id: number | null; score: number }[];
    chunks: { id: number; video_id: number; start_s: number; end_s: number; text: string; score: number }[];
  };
}

export interface Source {
  id: number;
  yt_channel_id: string;
  handle: string | null;
  title: string | null;
  videos_total: number;
}

export interface VideoPage {
  yt_video_id: string;
  title: string | null;
  url: string | null;
  ingest_status: string;
  segments: { start_s: number; end_s: number; text: string }[];
  units: { id: number; text: string; unit_type: string; topic_slug: string | null }[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const getHealth = () => get<Health>("/health");
export const getTree = () => get<TreeNode[]>("/kb/tree");
export const getTopic = (slug: string) => get<TopicDetail>(`/kb/topics/${encodeURIComponent(slug)}`);
export const getVersions = (slug: string) =>
  get<VersionInfo[]>(`/kb/topics/${encodeURIComponent(slug)}/versions`);
export const getVersion = (slug: string, v: number) =>
  get<Article>(`/kb/topics/${encodeURIComponent(slug)}/versions/${v}`);
export const getUnit = (id: number) => get<UnitProvenance>(`/kb/units/${id}`);
export const search = (q: string) => get<SearchResults>(`/kb/search?q=${encodeURIComponent(q)}`);
export const getSources = () => get<Source[]>("/kb/sources");
export const getVideo = (ytId: string) => get<VideoPage>(`/kb/videos/${encodeURIComponent(ytId)}`);

export interface ChatEvent {
  type: "session" | "token" | "done";
  session_id?: number;
  text?: string;
  sources?: { label: string; kind: string; ref: string }[];
}

export async function chatStream(
  message: string,
  sessionId: number | null,
  onEvent: (e: ChatEvent) => void,
): Promise<void> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (line.startsWith("data:")) {
        try {
          onEvent(JSON.parse(line.slice(5).trim()) as ChatEvent);
        } catch {
          /* ignore malformed frame */
        }
      }
    }
  }
}

export function youtubeUrl(ytVideoId: string, startS?: number): string {
  const base = `https://www.youtube.com/watch?v=${ytVideoId}`;
  return startS != null ? `${base}&t=${Math.floor(startS)}s` : base;
}

export function formatTs(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
