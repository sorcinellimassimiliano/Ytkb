import { useEffect, useState, type ReactNode } from "react";
import { getUnit, youtubeUrl, formatTs, type UnitProvenance } from "./api";

const unitCache = new Map<number, Promise<UnitProvenance>>();
function loadUnit(id: number): Promise<UnitProvenance> {
  if (!unitCache.has(id)) unitCache.set(id, getUnit(id));
  return unitCache.get(id)!;
}

function Citation({ id }: { id: number }) {
  const [unit, setUnit] = useState<UnitProvenance | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let alive = true;
    loadUnit(id)
      .then((u) => alive && setUnit(u))
      .catch(() => alive && setErr(true));
    return () => {
      alive = false;
    };
  }, [id]);

  if (err) return <sup className="cite cite-missing">[unit:{id}]</sup>;
  if (!unit || !unit.video) return <sup className="cite">[{id}]</sup>;

  const start = unit.chunks[0]?.start_s;
  const href = youtubeUrl(unit.video.yt_video_id, start);
  const label = start != null ? formatTs(start) : "fonte";
  const title = `${unit.video.title ?? unit.video.yt_video_id} — ${unit.text}`;
  return (
    <sup className="cite">
      <a href={href} target="_blank" rel="noreferrer" title={title}>
        ▶ {label}
      </a>
    </sup>
  );
}

const CITE = /\[unit:(\d+)\]/g;

function renderInline(text: string, keyBase: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  CITE.lastIndex = 0;
  let i = 0;
  while ((m = CITE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    nodes.push(<Citation key={`${keyBase}-c${i++}`} id={Number(m[1])} />);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let list: ReactNode[] = [];
  let key = 0;

  const flushList = () => {
    if (list.length) {
      blocks.push(<ul key={`ul-${key++}`}>{list}</ul>);
      list = [];
    }
  };

  for (const line of lines) {
    if (line.startsWith("## ")) {
      flushList();
      blocks.push(<h3 key={`h-${key++}`}>{renderInline(line.slice(3), `h${key}`)}</h3>);
    } else if (line.startsWith("# ")) {
      flushList();
      blocks.push(<h2 key={`h-${key++}`}>{renderInline(line.slice(2), `h${key}`)}</h2>);
    } else if (line.startsWith("- ")) {
      list.push(<li key={`li-${key++}`}>{renderInline(line.slice(2), `li${key}`)}</li>);
    } else if (line.trim() === "") {
      flushList();
    } else {
      flushList();
      blocks.push(<p key={`p-${key++}`}>{renderInline(line, `p${key}`)}</p>);
    }
  }
  flushList();
  return <div className="markdown">{blocks}</div>;
}
