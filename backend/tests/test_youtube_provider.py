from datetime import UTC

from ytkb.ingestion.providers.youtube import _channel_url, _parse_upload_date


def test_channel_url_variants():
    assert _channel_url("@creator") == "https://www.youtube.com/@creator"
    assert _channel_url("UC123abc") == "https://www.youtube.com/channel/UC123abc"
    assert _channel_url("creator") == "https://www.youtube.com/@creator"
    assert _channel_url("https://youtube.com/@x/") == "https://youtube.com/@x"


def test_parse_upload_date():
    dt = _parse_upload_date("20260101")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 1 and dt.day == 1
    assert dt.tzinfo == UTC
    assert _parse_upload_date(None) is None
    assert _parse_upload_date("garbage") is None
