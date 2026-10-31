"""Map diarization labels (SPEAKER_00, ...) to real names with one Claude call."""

from __future__ import annotations

import json
import logging
import re

from .config import Config
from .feed import Episode
from .turns import Turn, format_ts, speaker_stats

log = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 300_000  # ~75k tokens; a 2h episode is roughly 110k chars

SYSTEM_PROMPT = """You label speakers in podcast transcripts.
The transcript was produced by automatic speech recognition plus speaker diarization,
so each turn carries an anonymous label like SPEAKER_00. Your job is to work out which
real person each label is, using evidence in the text: people addressing each other by
name ("Chamath, what do you think?"), introductions ("welcome back, Nikesh"), the host
running the show, self-references, and topics each person is known for.

Rules:
- Regular hosts are listed below; guests are usually named in the episode title or description.
- Not every host appears in every episode. Do not assign a host who is not evidently speaking.
- Diarization sometimes splits one person into two labels; mapping two labels to the same
  name is allowed.
- A label with only a few words (ads, clips, crosstalk) may be "Unknown" if the evidence is thin.
- Spell names correctly even if the transcript misspells them.
- Respond with JSON only, no prose, in exactly this shape:
  {"mapping": {"SPEAKER_00": "Full Name", ...},
   "confidence": {"SPEAKER_00": "high|medium|low", ...},
   "notes": "one short sentence on anything uncertain"}"""


def _transcript_for_prompt(turns: list[Turn]) -> str:
    lines = [f"[{format_ts(t.start)}] {t.speaker}: {t.text}" for t in turns]
    text = "\n".join(lines)
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text
    # Keep the start (intros carry the most evidence) and the end.
    head = text[: int(MAX_TRANSCRIPT_CHARS * 0.7)]
    tail = text[-int(MAX_TRANSCRIPT_CHARS * 0.3):]
    return head + "\n[... middle of transcript omitted ...]\n" + tail


def build_user_prompt(episode: Episode, turns: list[Turn], hosts: list[str]) -> str:
    stats = speaker_stats(turns)
    stats_lines = "\n".join(
        f"- {label}: {s['words']} words, {s['turns']} turns, first speaks at "
        f"{format_ts(next(t.start for t in turns if t.speaker == label))}"
        for label, s in stats.items()
    )
    return (
        f"Podcast regular hosts: {', '.join(hosts)}\n"
        f"Episode title: {episode.title}\n"
        f"Episode description:\n{episode.description[:3000]}\n\n"
        f"Diarization labels found:\n{stats_lines}\n\n"
        f"Transcript:\n{_transcript_for_prompt(turns)}"
    )


def parse_mapping(text: str, labels: set[str]) -> tuple[dict[str, str], dict]:
    """Extract the JSON object from the model's reply; tolerate code fences / chatter."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object in model reply")
    data = json.loads(match.group(0))
    raw = data.get("mapping") or {}
    mapping = {
        label: name.strip()
        for label, name in raw.items()
        if label in labels and isinstance(name, str) and name.strip()
        and name.strip().lower() not in {"unknown", "unknown speaker", "n/a"}
    }
    return mapping, data


def map_speakers(episode: Episode, turns: list[Turn], cfg: Config) -> dict:
    """Return {"mapping": {...}, "confidence": {...}, "notes": str, "model": str}."""
    labels = {t.speaker for t in turns}
    if not cfg.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not set: speakers will stay as 'Speaker N'")
        return {"mapping": {}, "confidence": {}, "notes": "No API key; speakers not named.", "model": None}

    import anthropic

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    prompt = build_user_prompt(episode, turns, cfg.hosts)
    last_error: Exception | None = None
    for attempt in range(2):
        resp = client.messages.create(
            model=cfg.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        try:
            mapping, data = parse_mapping(reply, labels)
            log.info("Speaker mapping: %s", mapping)
            return {
                "mapping": mapping,
                "confidence": data.get("confidence") or {},
                "notes": data.get("notes") or "",
                "model": cfg.claude_model,
            }
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            log.warning("Could not parse mapping (attempt %d): %s", attempt + 1, exc)
    raise RuntimeError(f"Speaker mapping failed: {last_error}")
