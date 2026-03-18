from __future__ import annotations

from app.services.subtitles import segments_to_srt


def test_segments_to_srt_formats_and_orders() -> None:
    srt = segments_to_srt(
        [
            {"start": 0.0, "end": 1.2, "text": "Hello"},
            {"start": 1.2, "end": 2.0, "text": "World"},
        ]
    )
    assert "00:00:00,000 --> 00:00:01,200" in srt
    assert "Hello" in srt
    assert "00:00:01,200 --> 00:00:02,000" in srt
    assert "World" in srt

