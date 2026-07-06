"""APScheduler in-process scheduler for nightly channel scans."""

from __future__ import annotations

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ytkb.config import get_settings
from ytkb.db import repository as repo
from ytkb.db.base import get_sessionmaker
from ytkb.ingestion.providers.youtube import YouTubeProvider
from ytkb.ingestion.service import discover_videos, transcribe_pending
from ytkb.logging import configure_logging, get_logger

log = get_logger("scheduler")


async def scan_all_channels() -> None:
    """Discover new videos for every active channel and transcribe pending ones.

    Errors on a single channel are logged and skipped so one bad channel never
    aborts the nightly run.
    """
    settings = get_settings()
    provider = YouTubeProvider()
    async with get_sessionmaker()() as session:
        channels = await repo.list_active_channels(session)
        log.info("channel_scan_start", channels=len(channels))
        for channel in channels:
            try:
                new = await discover_videos(session, provider, channel)
                log.info("channel_scanned", channel=channel.yt_channel_id, new=len(new))
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("channel_scan_failed", channel=channel.yt_channel_id, error=str(exc))
                await session.rollback()
        try:
            n = await transcribe_pending(session, provider, languages=settings.transcript_languages)
            log.info("channel_scan_transcribed", count=n)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("scan_transcribe_failed", error=str(exc))
        await session.commit()


def build_scheduler() -> AsyncIOScheduler:
    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        scan_all_channels,
        CronTrigger.from_crontab(settings.channel_scan_cron),
        id="channel_scan",
        replace_existing=True,
    )
    return scheduler


def run_forever() -> None:
    configure_logging()
    settings = get_settings()
    if not settings.scheduler_enabled:
        log.info("scheduler_disabled")
        return
    scheduler = build_scheduler()
    scheduler.start()
    log.info("scheduler_started", cron=settings.channel_scan_cron)
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        scheduler.shutdown()
