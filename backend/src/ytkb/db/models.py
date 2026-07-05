"""ORM models — the two-layer schema (source layer + knowledge layer).

The schema mirrors migration 0001. Vector dimension is fixed there; the ORM
reads it from settings so application code stays consistent, but the migration
is the source of truth.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ytkb.config import get_settings
from ytkb.db.base import Base

EMBED_DIM = get_settings().embedding_dim


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class VideoKind(str, enum.Enum):
    video = "video"
    short = "short"


class IngestStatus(str, enum.Enum):
    pending = "pending"
    transcribed = "transcribed"
    no_transcript = "no_transcript"
    chunked = "chunked"
    embedded = "embedded"
    units_extracted = "units_extracted"
    assigned = "assigned"
    merged = "merged"
    error = "error"


class TranscriptSource(str, enum.Enum):
    yt_manual = "yt_manual"
    yt_auto = "yt_auto"
    whisper = "whisper"
    cloud_asr = "cloud_asr"


class UnitType(str, enum.Enum):
    technique = "technique"
    claim = "claim"
    opinion = "opinion"
    tool = "tool"
    workflow = "workflow"
    example = "example"
    definition = "definition"


class Assignment(str, enum.Enum):
    auto = "auto"
    llm = "llm"
    manual = "manual"
    pending = "pending"


class TopicStatus(str, enum.Enum):
    active = "active"
    proposed = "proposed"
    merged_into = "merged_into"


def _ts() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# --------------------------------------------------------------------------
# Source layer
# --------------------------------------------------------------------------
class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    yt_channel_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    handle: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(512))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _ts()

    videos: Mapped[list[Video]] = relationship(back_populates="channel")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    yt_video_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(1024))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_s: Mapped[int | None] = mapped_column(Integer)
    kind: Mapped[VideoKind] = mapped_column(
        Enum(VideoKind, name="video_kind"), default=VideoKind.video, nullable=False
    )
    url: Mapped[str | None] = mapped_column(String(512))
    ingest_status: Mapped[IngestStatus] = mapped_column(
        Enum(IngestStatus, name="ingest_status"), default=IngestStatus.pending, nullable=False
    )
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    channel: Mapped[Channel] = relationship(back_populates="videos")
    transcripts: Mapped[list[Transcript]] = relationship(back_populates="video")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="video")


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    language: Mapped[str | None] = mapped_column(String(16))
    source: Mapped[TranscriptSource] = mapped_column(
        Enum(TranscriptSource, name="transcript_source"), nullable=False
    )
    raw_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _ts()

    video: Mapped[Video] = relationship(back_populates="transcripts")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("video_id", "start_s", name="uq_chunk_video_start"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    # tsv is a GENERATED column created in the migration (FTS, GIN index).
    created_at: Mapped[datetime] = _ts()

    video: Mapped[Video] = relationship(back_populates="chunks")


# --------------------------------------------------------------------------
# Knowledge layer
# --------------------------------------------------------------------------
class KnowledgeUnit(Base):
    __tablename__ = "knowledge_units"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False)
    unit_type: Mapped[UnitType] = mapped_column(Enum(UnitType, name="unit_type"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), nullable=True
    )
    assignment: Mapped[Assignment] = mapped_column(
        Enum(Assignment, name="assignment"), default=Assignment.pending, nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = _ts()

    topic: Mapped[Topic | None] = relationship(back_populates="units")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), nullable=True
    )
    centroid: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM))
    status: Mapped[TopicStatus] = mapped_column(
        Enum(TopicStatus, name="topic_status"), default=TopicStatus.proposed, nullable=False
    )
    merged_into_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), nullable=True
    )
    units_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _ts()

    units: Mapped[list[KnowledgeUnit]] = relationship(
        back_populates="topic", foreign_keys="KnowledgeUnit.topic_id"
    )
    articles: Mapped[list[TopicArticle]] = relationship(back_populates="topic")


class TopicArticle(Base):
    """Append-only. The row with the highest version is the current article."""

    __tablename__ = "topic_articles"
    __table_args__ = (UniqueConstraint("topic_id", "version", name="uq_article_topic_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_md: Mapped[str] = mapped_column(Text, nullable=False)
    units_included: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False)
    change_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _ts()

    topic: Mapped[Topic] = relationship(back_populates="articles")


class TopicRelation(Base):
    __tablename__ = "topic_relations"
    __table_args__ = (UniqueConstraint("topic_a", "topic_b", "kind", name="uq_topic_relation"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_a: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    topic_b: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32), nullable=False)


# --------------------------------------------------------------------------
# Chat (optional layer)
# --------------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = _ts()

    messages: Mapped[list[ChatMessage]] = relationship(back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _ts()

    session: Mapped[ChatSession] = relationship(back_populates="messages")
