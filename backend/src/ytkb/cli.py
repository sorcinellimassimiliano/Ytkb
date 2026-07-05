"""Operational CLI (typer). Wraps the async pipeline services."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import typer

from ytkb.config import get_settings
from ytkb.db.base import get_sessionmaker
from ytkb.ingestion.providers.youtube import YouTubeProvider
from ytkb.ingestion.service import (
    discover_videos,
    register_channel,
    transcribe_pending,
)
from ytkb.logging import configure_logging

app = typer.Typer(help="YT-KB operational CLI", no_args_is_help=True)

T = TypeVar("T")


def _run(coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
    configure_logging()
    try:
        return asyncio.run(coro())
    except Exception as exc:
        # Surface network / provider failures as a clean CLI error, not a
        # traceback. Rate limiting from datacenter IPs is expected here.
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


@app.command()
def add_channel(handle_or_id: str) -> None:
    """Register a YouTube channel by @handle, UC-id or URL."""

    async def _do() -> None:
        provider = YouTubeProvider()
        async with get_sessionmaker()() as session:
            channel = await register_channel(session, provider, handle_or_id)
            await session.commit()
            typer.echo(f"Channel: {channel.title} ({channel.yt_channel_id})")

    _run(_do)


@app.command()
def ingest(
    handle_or_id: str = typer.Argument(None, help="Channel to ingest; omit for all active"),
    limit: int = typer.Option(None, help="Max videos to discover per channel"),
    transcribe: bool = typer.Option(True, help="Fetch transcripts for pending videos"),
    retry_errors: bool = typer.Option(False, help="Also retry videos in 'error' state"),
) -> None:
    """Discover videos for a channel and fetch transcripts."""

    async def _do() -> None:
        settings = get_settings()
        provider = YouTubeProvider()
        async with get_sessionmaker()() as session:
            if handle_or_id:
                channel = await register_channel(session, provider, handle_or_id)
                new = await discover_videos(session, provider, channel, limit=limit)
                typer.echo(f"Discovered {len(new)} new videos.")
            if transcribe:
                n = await transcribe_pending(
                    session,
                    provider,
                    languages=settings.transcript_languages,
                    limit=limit,
                    include_errors=retry_errors,
                )
                typer.echo(f"Transcribed {n} videos.")
            await session.commit()

    _run(_do)


@app.command()
def scan() -> None:
    """Run one channel scan now (discover + transcribe for all active channels)."""
    from ytkb.ingestion.scheduler import scan_all_channels

    async def _do() -> None:
        await scan_all_channels()
        typer.echo("Scan complete.")

    _run(_do)


@app.command()
def index(limit: int = typer.Option(None, help="Max transcribed videos to process")) -> None:
    """Chunk + embed all transcribed videos (transcribed → chunked → embedded)."""
    from ytkb.indexing.pipeline import index_transcribed

    async def _do() -> None:
        async with get_sessionmaker()() as session:
            totals = await index_transcribed(session, limit=limit)
            await session.commit()
            typer.echo(
                f"Indexed {totals['videos']} videos, "
                f"{totals['chunks']} chunks, {totals['embedded']} embeddings."
            )

    _run(_do)


@app.command()
def reindex_video(yt_video_id: str) -> None:
    """Force a clean re-chunk + re-embed of a single video."""
    from ytkb.indexing.pipeline import reindex_video as _reindex

    async def _do() -> None:
        async with get_sessionmaker()() as session:
            res = await _reindex(session, yt_video_id)
            await session.commit()
            typer.echo(f"{res['video']}: {res['chunks']} chunks, {res['embedded']} embeddings.")

    _run(_do)


@app.command()
def extract_units(
    limit: int = typer.Option(None, help="Max embedded videos to process"),
) -> None:
    """Extract knowledge units from embedded videos (embedded → units_extracted)."""
    from ytkb.knowledge.extraction import extract_pending

    async def _do() -> None:
        async with get_sessionmaker()() as session:
            totals = await extract_pending(session, limit=limit)
            await session.commit()
            typer.echo(f"Extracted {totals['units']} units from {totals['videos']} videos.")

    _run(_do)


@app.command()
def assign(limit: int = typer.Option(None, help="Max pending units to assign")) -> None:
    """Assign pending knowledge units to topics (matcher + arbitration)."""
    from ytkb.knowledge.assignment import assign_pending_units

    async def _do() -> None:
        async with get_sessionmaker()() as session:
            res = await assign_pending_units(session, limit=limit)
            await session.commit()
            typer.echo(
                f"Assigned {res['units_assigned']} units, created {res['topics_created']} topics."
            )

    _run(_do)


@app.command()
def search(
    query: str,
    mode: str = typer.Option("hybrid", help="hybrid | semantic | fts"),
    limit: int = typer.Option(10),
) -> None:
    """Search transcript chunks from the CLI (no LLM)."""
    from ytkb.db import search as search_svc
    from ytkb.indexing.embeddings import get_embedding_client

    async def _do() -> None:
        async with get_sessionmaker()() as session:
            if mode == "fts":
                hits = await search_svc.fts_chunks(session, query, limit=limit)
            else:
                vec = await get_embedding_client().embed_one(query)
                if mode == "semantic":
                    hits = await search_svc.semantic_chunks(session, vec, limit=limit)
                else:
                    hits = await search_svc.hybrid_chunks(session, query, vec, limit=limit)
            for h in hits:
                typer.echo(f"[{h.score:.4f}] v{h.video_id} {h.start_s:.0f}s  {h.text[:80]}")

    _run(_do)


@app.command()
def status() -> None:
    """Show pipeline counts by ingest status."""
    from sqlalchemy import func, select

    from ytkb.db.models import Video

    async def _do() -> None:
        async with get_sessionmaker()() as session:
            rows = await session.execute(
                select(Video.ingest_status, func.count(Video.id)).group_by(Video.ingest_status)
            )
            for st, count in rows:
                typer.echo(f"{st.value:16s} {count}")

    _run(_do)


@app.command()
def scheduler() -> None:
    """Run the in-process scheduler (nightly channel scans)."""
    from ytkb.ingestion.scheduler import run_forever

    run_forever()


if __name__ == "__main__":
    app()
