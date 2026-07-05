"""Integration tests for topic assignment (matcher) against real Postgres.

Embeddings are hand-built unit vectors so cosine similarity — and therefore the
θ_high / θ_low / band decisions — are exact and deterministic.
"""

from __future__ import annotations

import math

from sqlalchemy import func, select

from ytkb.config import Settings
from ytkb.db.models import Assignment, IngestStatus, KnowledgeUnit, Topic, TopicStatus
from ytkb.knowledge.assignment import assign_pending_units

from .conftest import requires_db
from .fakes import seed_bare_video, seed_unit

pytestmark = requires_db

DIM = 1536


def angle_vec(cos: float) -> list[float]:
    """Unit vector whose cosine similarity to e0=[1,0,...] is exactly `cos`."""
    v = [0.0] * DIM
    v[0] = cos
    v[1] = math.sqrt(max(0.0, 1.0 - cos * cos))
    return v


E0 = angle_vec(1.0)  # reference direction
ORTHO = angle_vec(0.0)  # similarity 0 to E0

SETTINGS = Settings(theta_high=0.82, theta_low=0.62, topic_promote_after_units=3)


async def test_first_unit_creates_proposed_topic(session):
    video = await seed_bare_video(session, yt_video_id="a1")
    unit = await seed_unit(session, video=video, text="unit one", embedding=E0)
    res = await assign_pending_units(session, settings=SETTINGS)
    assert res["units_assigned"] == 1
    assert res["topics_created"] == 1
    await session.refresh(unit)
    assert unit.topic_id is not None
    topic = await session.get(Topic, unit.topic_id)
    assert topic.status == TopicStatus.proposed
    assert topic.units_count == 1


async def test_similar_units_land_in_same_topic(session):
    video = await seed_bare_video(session, yt_video_id="a2")
    await seed_unit(session, video=video, text="u1", embedding=E0)
    await seed_unit(session, video=video, text="u2", embedding=angle_vec(0.95))
    await assign_pending_units(session, settings=SETTINGS)
    topics = await session.scalar(select(func.count(Topic.id)))
    assert topics == 1  # 0.95 ≥ θ_high → same topic


async def test_dissimilar_units_create_separate_topics(session):
    video = await seed_bare_video(session, yt_video_id="a3")
    await seed_unit(session, video=video, text="u1", embedding=E0)
    await seed_unit(session, video=video, text="u2", embedding=ORTHO)
    await assign_pending_units(session, settings=SETTINGS)
    topics = await session.scalar(select(func.count(Topic.id)))
    assert topics == 2  # similarity 0 < θ_low → new topic


async def test_band_arbiter_assigns_when_above_midpoint(session):
    # midpoint = 0.72. sim 0.78 is in-band and above midpoint → assign existing.
    video = await seed_bare_video(session, yt_video_id="a4")
    await seed_unit(session, video=video, text="u1", embedding=E0)
    u2 = await seed_unit(session, video=video, text="u2", embedding=angle_vec(0.78))
    await assign_pending_units(session, settings=SETTINGS)
    topics = await session.scalar(select(func.count(Topic.id)))
    assert topics == 1
    await session.refresh(u2)
    assert u2.assignment == Assignment.llm  # decided by arbitration


async def test_band_arbiter_proposes_new_below_midpoint(session):
    # sim 0.65 is in-band but below midpoint 0.72 → new topic.
    video = await seed_bare_video(session, yt_video_id="a5")
    await seed_unit(session, video=video, text="u1", embedding=E0)
    await seed_unit(session, video=video, text="u2", embedding=angle_vec(0.65))
    await assign_pending_units(session, settings=SETTINGS)
    topics = await session.scalar(select(func.count(Topic.id)))
    assert topics == 2


async def test_topic_promoted_after_threshold(session):
    video = await seed_bare_video(session, yt_video_id="a6")
    for i in range(3):
        await seed_unit(session, video=video, text=f"u{i}", embedding=angle_vec(0.99))
    await assign_pending_units(session, settings=SETTINGS)
    topic = (await session.execute(select(Topic))).scalars().first()
    assert topic.units_count == 3
    assert topic.status == TopicStatus.active  # promoted at 3


async def test_rerun_is_idempotent_and_advances_video(session):
    video = await seed_bare_video(session, yt_video_id="a7")
    await seed_unit(session, video=video, text="u1", embedding=E0)
    await seed_unit(session, video=video, text="u2", embedding=angle_vec(0.9))
    first = await assign_pending_units(session, settings=SETTINGS)
    assert first["units_assigned"] == 2
    await session.refresh(video)
    assert video.ingest_status == IngestStatus.assigned

    second = await assign_pending_units(session, settings=SETTINGS)
    assert second["units_assigned"] == 0  # nothing pending
    pending = await session.scalar(
        select(func.count(KnowledgeUnit.id)).where(KnowledgeUnit.assignment == Assignment.pending)
    )
    assert pending == 0
