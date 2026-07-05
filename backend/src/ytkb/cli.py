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
    return asyncio.run(coro())


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
                    session, provider, languages=settings.transcript_languages, limit=limit
                )
                typer.echo(f"Transcribed {n} videos.")
            await session.commit()

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
