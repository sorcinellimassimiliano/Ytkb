"""Integration tests for incremental merge — the Phase 4 acceptance:
'3 videos → 1 article that grows', versions kept, citations valid, no orphans."""

from __future__ import annotations

import math

from sqlalchemy import func, select

from ytkb.config import Settings
from ytkb.db.models import (
    IngestStatus,
    Topic,
    TopicArticle,
    TopicStatus,
    UnitType,
)
from ytkb.knowledge.assignment import assign_pending_units
from ytkb.knowledge.merge import (
    cited_unit_ids,
    merge_dirty_topics,
    merge_topic,
    rebuild_topic,
)
from ytkb.knowledge.review import find_similar_topics, merge_topics

from .conftest import requires_db
from .fakes import seed_bare_video, seed_unit

pytestmark = requires_db

DIM = 1536


def vec(cos: float) -> list[float]:
    v = [0.0] * DIM
    v[0] = cos
    v[1] = math.sqrt(max(0.0, 1.0 - cos * cos))
    return v


E0 = vec(1.0)
# High promote threshold so topics stay 'proposed' and we isolate merge behaviour.
SETTINGS = Settings(theta_high=0.82, theta_low=0.62, topic_promote_after_units=100)


async def _ingest_video_units(session, idx: int):
    video = await seed_bare_video(
        session, yt_video_id=f"m{idx}", status=IngestStatus.units_extracted
    )
    await seed_unit(
        session,
        video=video,
        text=f"Come configurare il context nel modo {idx}",
        embedding=E0,
        unit_type=UnitType.technique,
    )
    await seed_unit(
        session,
        video=video,
        text=f"Lo strumento prompt caching versione {idx}",
        embedding=E0,
        unit_type=UnitType.tool,
    )
    return video


async def test_three_videos_grow_one_article(session):
    expected = [(1, 2), (2, 4), (3, 6)]  # (version, units_included) after each video
    videos = []
    for i in range(3):
        videos.append(await _ingest_video_units(session, i))
        await assign_pending_units(session, settings=SETTINGS)
        await merge_dirty_topics(session, settings=SETTINGS)

        topic_count = await session.scalar(select(func.count(Topic.id)))
        assert topic_count == 1, "all units share one topic"
        topic = (await session.execute(select(Topic))).scalars().one()
        latest = (
            (
                await session.execute(
                    select(TopicArticle)
                    .where(TopicArticle.topic_id == topic.id)
                    .order_by(TopicArticle.version.desc())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        assert (latest.version, len(latest.units_included)) == expected[i]

    # Every unit in the final article is cited — no orphans, nothing dropped.
    final = (
        (await session.execute(select(TopicArticle).order_by(TopicArticle.version.desc()).limit(1)))
        .scalars()
        .first()
    )
    assert set(final.units_included) == cited_unit_ids(final.content_md)

    # All three source videos advanced to merged.
    for v in videos:
        await session.refresh(v)
        assert v.ingest_status == IngestStatus.merged


async def test_merge_is_idempotent_without_new_units(session):
    await _ingest_video_units(session, 0)
    await assign_pending_units(session, settings=SETTINGS)
    first = await merge_dirty_topics(session, settings=SETTINGS)
    assert first["topics_merged"] == 1
    second = await merge_dirty_topics(session, settings=SETTINGS)
    assert second["topics_merged"] == 0  # nothing dirty
    versions = await session.scalar(select(func.count(TopicArticle.id)))
    assert versions == 1


async def test_rebuild_topic_creates_new_version_from_all_units(session):
    await _ingest_video_units(session, 0)
    await assign_pending_units(session, settings=SETTINGS)
    await merge_dirty_topics(session, settings=SETTINGS)
    topic = (await session.execute(select(Topic))).scalars().one()

    rebuilt = await rebuild_topic(session, topic.slug)
    assert rebuilt.version == 2
    assert len(rebuilt.units_included) == 2
    assert cited_unit_ids(rebuilt.content_md) == set(rebuilt.units_included)


async def test_merge_topic_returns_none_when_clean(session):
    await _ingest_video_units(session, 0)
    await assign_pending_units(session, settings=SETTINGS)
    topic = (await session.execute(select(Topic))).scalars().one()
    await merge_topic(session, topic, settings=SETTINGS)
    # Second call with no new units → None.
    assert await merge_topic(session, topic, settings=SETTINGS) is None


async def test_topics_review_and_manual_merge(session):
    # Two distinct topics with identical centroids → review proposes a merge.
    video = await seed_bare_video(session, yt_video_id="rev", status=IngestStatus.units_extracted)
    ta = Topic(slug="ta", title="TA", status=TopicStatus.active, units_count=1, centroid=E0)
    tb = Topic(slug="tb", title="TB", status=TopicStatus.active, units_count=1, centroid=E0)
    session.add_all([ta, tb])
    await session.flush()
    u = await seed_unit(session, video=video, text="unità di tb", embedding=E0)
    u.topic_id = tb.id
    await session.flush()

    settings = Settings(topic_similarity_review_threshold=0.9)
    pairs = await find_similar_topics(session, settings=settings)
    assert any({p.topic_a_slug, p.topic_b_slug} == {"ta", "tb"} for p in pairs)

    res = await merge_topics(session, "tb", "ta")
    assert res["units_remapped"] == 1
    await session.refresh(u)
    await session.refresh(tb)
    assert u.topic_id == ta.id
    assert tb.status == TopicStatus.merged_into
    assert tb.merged_into_id == ta.id
