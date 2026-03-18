from __future__ import annotations

from dataclasses import dataclass


def _fmt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        if end <= start:
            end = start + 0.8
        lines.append(str(idx))
        lines.append(f"{_fmt_timestamp(start)} --> {_fmt_timestamp(end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class SubtitleSpec:
    srt_text: str

