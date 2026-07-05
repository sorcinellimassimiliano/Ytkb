"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ytkb import __version__
from ytkb.api.routers import admin, chat, health, kb
from ytkb.config import get_settings
from ytkb.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="YT-KB",
        version=__version__,
        summary="Memoria totale per argomento da contenuti YouTube",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(kb.router)
    app.include_router(admin.router)
    if settings.chat_enabled:
        app.include_router(chat.router)

    return app


app = create_app()
