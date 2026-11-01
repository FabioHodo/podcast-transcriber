"""Key takeaways: one Claude call over the named transcript -> 10-20 attributed bullet points."""

from __future__ import annotations

import json
import logging
import re

from .config import Config
from .feed import Episode
from .turns import Turn, format_ts

log = logging.getLogger(__name__)

MIN_TAKEAWAYS, MAX_TAKEAWAYS = 10, 20
TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")

SYSTEM_PROMPT = f"""You summarize podcast episodes for busy listeners.
You get a machine transcript whose speaker names were inferred automatically.

Pick the {MIN_TAKEAWAYS}-{MAX_TAKEAWAYS} most important takeaways of the episode: claims, predictions,
numbers, disagreements and conclusions a listener would want to know. Skip ads, intros, banter and
housekeeping. Order them as they occur in the episode.

For each takeaway:
- "point": one plain sentence, specific (keep names, figures, companies), no hype.
- "speaker": the speaker who made the point, exactly as named in the transcript. If it came out of
  an exchange between several people, name the one who made the core claim.
- "timestamp": the HH:MM:SS where that speaker starts making the point, copied from the transcript."""

SCHEMA = {
    "type": "object",
    "properties": {
        "takeaways": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "point": {"type": "string"},
                },
                "required": ["speaker", "timestamp", "point"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["takeaways"],
    "additionalProperties": False,
}


def build_user_prompt(episode: Episode, turns: list[Turn]) -> str:
    transcript = "\n".join(f"[{format_ts(t.start)}] {t.speaker}: {t.text}" for t in turns)
    return (
        f"Episode title: {episode.title}\n"
        f"Episode description:\n{episode.description[:3000]}\n\n"
        f"Transcript:\n{transcript}"
    )


def parse_takeaways(text: str, speakers: set[str]) -> list[dict]:
    """Validate the model's JSON; drop malformed items and cap the list at MAX_TAKEAWAYS."""
    out = []
    for item in json.loads(text).get("takeaways") or []:
        point = (item.get("point") or "").strip()
        if not point:
            continue
        speaker = (item.get("speaker") or "").strip()
        ts = (item.get("timestamp") or "").strip()
        out.append({
            "speaker": speaker if speaker in speakers else "",
            "timestamp": ts if TS_RE.match(ts) else "",
            "point": point,
        })
    return out[:MAX_TAKEAWAYS]


def summarize(episode: Episode, turns: list[Turn], cfg: Config, client=None) -> dict:
    """Return {"takeaways": [{speaker, timestamp, point}, ...], "model": str | None}."""
    if not cfg.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not set: no key takeaways")
        return {"takeaways": [], "model": None}

    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    resp = client.beta.messages.create(
        model=cfg.summary_model,
        max_tokens=16000,
        # If a safety classifier declines, the API retries on a suitable model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(episode, turns)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Takeaways request was declined: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Takeaways reply was cut off at max_tokens")
    reply = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    takeaways = parse_takeaways(reply, {t.speaker for t in turns})
    log.info("Generated %d key takeaways with %s", len(takeaways), resp.model)
    return {"takeaways": takeaways, "model": resp.model}
