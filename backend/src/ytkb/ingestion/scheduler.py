"""APScheduler in-process scheduler for nightly channel scans."""

from __future__ import annotations

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ytkb.config import get_settings
from ytkb.logging import configure_logging, get_logger

log = get_logger("scheduler")


async def scan_all_channels() -> None:
    """Placeholder scan job — Phase 1 wires channel discovery + transcription."""
    log.info("channel_scan_tick")


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
