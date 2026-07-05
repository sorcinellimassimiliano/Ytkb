const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export interface Health {
  status: string;
  version: string;
  chat_enabled: boolean;
  embedding_dim: number;
  db: string;
  extensions: Record<string, boolean>;
}

export interface Topic {
  id: number;
  slug: string;
  title: string;
  summary: string | null;
  parent_id: number | null;
  status: string;
  units_count: number;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const getHealth = () => get<Health>("/health");
export const listTopics = () => get<Topic[]>("/kb/topics");
