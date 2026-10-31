"""Turn WhisperX segments into speaker turns, and relabel speakers with names."""

from __future__ import annotations

from dataclasses import dataclass, asdict

UNKNOWN = "UNKNOWN"


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _units(segments: list[dict]):
    """Yield (speaker, start, end, text) at word level when possible, else per segment."""
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                text = (w.get("word") or "").strip()
                if text:
                    yield w.get("speaker"), w.get("start"), w.get("end"), text
        else:
            text = (seg.get("text") or "").strip()
            if text:
                yield seg.get("speaker"), seg.get("start"), seg.get("end"), text


def build_turns(segments: list[dict], min_switch_words: int = 2) -> list[Turn]:
    """Group consecutive words by speaker.

    Words with no speaker (common for numbers or words diarization missed) inherit
    the current speaker. A speaker change shorter than `min_switch_words` words,
    sandwiched between the same speaker, is treated as diarization noise and
    folded back in (e.g. a stray "yeah" mislabelled mid-sentence).
    """
    raw: list[Turn] = []
    current: Turn | None = None
    last_time = 0.0
    for speaker, start, end, text in _units(segments):
        start = start if start is not None else last_time
        end = end if end is not None else start
        last_time = end
        speaker = speaker or (current.speaker if current else UNKNOWN)
        if current and speaker == current.speaker:
            current.text += " " + text
            current.end = end
        else:
            current = Turn(speaker, start, end, text)
            raw.append(current)

    # Fold tiny interruptions A [B: 1 word] A back into A.
    cleaned: list[Turn] = []
    i = 0
    while i < len(raw):
        turn = raw[i]
        if (
            cleaned
            and i + 1 < len(raw)
            and len(turn.text.split()) < min_switch_words
            and cleaned[-1].speaker == raw[i + 1].speaker
        ):
            prev, nxt = cleaned[-1], raw[i + 1]
            prev.text += " " + turn.text + " " + nxt.text
            prev.end = nxt.end
            i += 2
            continue
        cleaned.append(turn)
        i += 1
    return merge_adjacent(cleaned)


def merge_adjacent(turns: list[Turn]) -> list[Turn]:
    merged: list[Turn] = []
    for t in turns:
        if merged and merged[-1].speaker == t.speaker:
            merged[-1].text += " " + t.text
            merged[-1].end = t.end
        else:
            merged.append(Turn(t.speaker, t.start, t.end, t.text))
    return merged


def speaker_stats(turns: list[Turn]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for t in turns:
        s = stats.setdefault(t.speaker, {"seconds": 0.0, "words": 0, "turns": 0})
        s["seconds"] += max(0.0, t.end - t.start)
        s["words"] += len(t.text.split())
        s["turns"] += 1
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["words"]))


def apply_names(turns: list[Turn], mapping: dict[str, str]) -> list[Turn]:
    """Replace diarization labels with names; unmapped labels become 'Speaker N'."""
    fallback: dict[str, str] = {}
    out = []
    for t in turns:
        name = mapping.get(t.speaker)
        if not name:
            name = fallback.setdefault(t.speaker, f"Speaker {len(fallback) + 1}")
        out.append(Turn(name, t.start, t.end, t.text))
    # Two labels mapped to the same person (diarization split them) -> merge runs.
    return merge_adjacent(out)
