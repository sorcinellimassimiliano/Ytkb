from ytkb.config import Settings


def test_defaults():
    s = Settings()
    assert s.embedding_dim == 1536
    assert s.chat_enabled is False
    assert s.transcript_languages == ["it", "en"]
    assert 0.0 < s.theta_low < s.theta_high < 1.0


def test_sync_database_url_uses_psycopg():
    s = Settings(database_url="postgresql+asyncpg://u:p@h:5432/db")
    assert s.sync_database_url == "postgresql+psycopg://u:p@h:5432/db"
