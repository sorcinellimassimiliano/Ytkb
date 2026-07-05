"""Initial complete schema — source layer + knowledge layer.

The whole schema lands in one migration (per project plan): channels, videos,
transcripts, chunks (pgvector + FTS), knowledge_units, topics, topic_articles,
topic_relations, chat. Embedding dimension is FIXED to 1536 here.

Revision ID: 0001
Revises:
Create Date: 2026-07-05

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = 1536


def upgrade() -> None:
    # --- Extensions (idempotent; also created by db/init script) ----------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # --- Enums ------------------------------------------------------------
    op.execute("CREATE TYPE video_kind AS ENUM ('video','short')")
    op.execute(
        "CREATE TYPE ingest_status AS ENUM "
        "('pending','transcribed','no_transcript','chunked','embedded',"
        "'units_extracted','assigned','merged','error')"
    )
    op.execute(
        "CREATE TYPE transcript_source AS ENUM "
        "('yt_manual','yt_auto','whisper','cloud_asr')"
    )
    op.execute(
        "CREATE TYPE unit_type AS ENUM "
        "('technique','claim','opinion','tool','workflow','example','definition')"
    )
    op.execute("CREATE TYPE assignment AS ENUM ('auto','llm','manual','pending')")
    op.execute("CREATE TYPE topic_status AS ENUM ('active','proposed','merged_into')")

    # --- Source layer -----------------------------------------------------
    op.execute(
        """
        CREATE TABLE channels (
            id SERIAL PRIMARY KEY,
            yt_channel_id VARCHAR(64) UNIQUE NOT NULL,
            handle VARCHAR(255),
            title VARCHAR(512),
            active BOOLEAN NOT NULL DEFAULT TRUE,
            last_checked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE videos (
            id SERIAL PRIMARY KEY,
            yt_video_id VARCHAR(32) UNIQUE NOT NULL,
            channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            title VARCHAR(1024),
            published_at TIMESTAMPTZ,
            duration_s INTEGER,
            kind video_kind NOT NULL DEFAULT 'video',
            url VARCHAR(512),
            ingest_status ingest_status NOT NULL DEFAULT 'pending',
            error_msg TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_videos_channel_id ON videos(channel_id)")
    op.execute("CREATE INDEX ix_videos_ingest_status ON videos(ingest_status)")

    op.execute(
        """
        CREATE TABLE transcripts (
            id SERIAL PRIMARY KEY,
            video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            language VARCHAR(16),
            source transcript_source NOT NULL,
            raw_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_transcripts_video_id ON transcripts(video_id)")

    op.execute(
        f"""
        CREATE TABLE chunks (
            id BIGSERIAL PRIMARY KEY,
            video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            start_s DOUBLE PRECISION NOT NULL,
            end_s DOUBLE PRECISION NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER,
            embedding vector({EMBED_DIM}),
            tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('italian', coalesce(text, '')) ||
                to_tsvector('english', coalesce(text, ''))
            ) STORED,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_chunk_video_start UNIQUE (video_id, start_s)
        )
        """
    )
    op.execute("CREATE INDEX ix_chunks_video_id ON chunks(video_id)")
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING GIN(tsv)")
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX ix_chunks_text_trgm ON chunks USING GIN(text gin_trgm_ops)")

    # --- Topics (declared before knowledge_units for the FK) --------------
    op.execute(
        f"""
        CREATE TABLE topics (
            id SERIAL PRIMARY KEY,
            slug VARCHAR(255) UNIQUE NOT NULL,
            title VARCHAR(512) NOT NULL,
            summary TEXT,
            parent_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
            centroid vector({EMBED_DIM}),
            status topic_status NOT NULL DEFAULT 'proposed',
            merged_into_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
            units_count INTEGER NOT NULL DEFAULT 0,
            last_merged_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_topics_parent_id ON topics(parent_id)")
    op.execute("CREATE INDEX ix_topics_status ON topics(status)")
    op.execute(
        "CREATE INDEX ix_topics_centroid ON topics "
        "USING hnsw (centroid vector_cosine_ops)"
    )

    # --- Knowledge layer --------------------------------------------------
    op.execute(
        f"""
        CREATE TABLE knowledge_units (
            id BIGSERIAL PRIMARY KEY,
            content_hash VARCHAR(64) UNIQUE NOT NULL,
            video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            chunk_ids BIGINT[] NOT NULL,
            unit_type unit_type NOT NULL,
            text TEXT NOT NULL,
            embedding vector({EMBED_DIM}),
            topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
            assignment assignment NOT NULL DEFAULT 'pending',
            confidence DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('italian', coalesce(text, '')) ||
                to_tsvector('english', coalesce(text, ''))
            ) STORED
        )
        """
    )
    op.execute("CREATE INDEX ix_units_video_id ON knowledge_units(video_id)")
    op.execute("CREATE INDEX ix_units_topic_id ON knowledge_units(topic_id)")
    op.execute("CREATE INDEX ix_units_assignment ON knowledge_units(assignment)")
    op.execute("CREATE INDEX ix_units_tsv ON knowledge_units USING GIN(tsv)")
    op.execute(
        "CREATE INDEX ix_units_embedding ON knowledge_units "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.execute(
        """
        CREATE TABLE topic_articles (
            id SERIAL PRIMARY KEY,
            topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
            version INTEGER NOT NULL,
            content_md TEXT NOT NULL,
            units_included BIGINT[] NOT NULL,
            change_summary TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_article_topic_version UNIQUE (topic_id, version)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_articles_tsv ON topic_articles USING GIN("
        "to_tsvector('italian', coalesce(content_md, '')))"
    )

    op.execute(
        """
        CREATE TABLE topic_relations (
            id SERIAL PRIMARY KEY,
            topic_a INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
            topic_b INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
            kind VARCHAR(32) NOT NULL,
            CONSTRAINT uq_topic_relation UNIQUE (topic_a, topic_b, kind)
        )
        """
    )

    # --- Chat -------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE chat_sessions (
            id SERIAL PRIMARY KEY,
            title VARCHAR(512),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chat_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            sources JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_chat_messages_session_id ON chat_messages(session_id)")


def downgrade() -> None:
    for table in (
        "chat_messages",
        "chat_sessions",
        "topic_relations",
        "topic_articles",
        "knowledge_units",
        "chunks",
        "transcripts",
        "videos",
        "topics",
        "channels",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    for enum in (
        "topic_status",
        "assignment",
        "unit_type",
        "transcript_source",
        "ingest_status",
        "video_kind",
    ):
        op.execute(f"DROP TYPE IF EXISTS {enum}")
