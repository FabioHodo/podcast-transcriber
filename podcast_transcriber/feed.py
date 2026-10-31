"""Find the podcast RSS feed and parse its episodes."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

USER_AGENT = "allin-transcripts/1.0 (personal podcast transcription)"
ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"


@dataclass
class Episode:
    guid: str
    title: str
    published: str  # ISO 8601
    audio_url: str
    link: str | None
    description: str
    duration: str | None

    @property
    def date(self) -> str:
        return self.published[:10]

    @property
    def slug(self) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")[:60].rstrip("-")
        return f"{self.date}-{base or 'episode'}"

    def to_dict(self) -> dict:
        return asdict(self)


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def resolve_feed_url(apple_id: str) -> str:
    """Ask Apple's public lookup API for the show's RSS feed URL."""
    data = json.loads(http_get(f"https://itunes.apple.com/lookup?id={apple_id}"))
    results = data.get("results") or []
    if not results or not results[0].get("feedUrl"):
        raise RuntimeError(f"Apple lookup returned no feedUrl for id {apple_id}")
    return results[0]["feedUrl"]


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _parse_date(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def parse_feed(xml_bytes: bytes) -> list[Episode]:
    """Parse an RSS 2.0 podcast feed; newest episode first."""
    root = ET.fromstring(xml_bytes)
    episodes: list[Episode] = []
    for item in root.iter("item"):
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url") if enclosure is not None else None
        if not audio_url:
            continue
        title = (item.findtext("title") or "Untitled").strip()
        guid = (item.findtext("guid") or audio_url).strip()
        desc = item.findtext("description") or item.findtext(f"{ITUNES_NS}summary") or ""
        episodes.append(
            Episode(
                guid=guid,
                title=title,
                published=_parse_date(item.findtext("pubDate")),
                audio_url=audio_url.strip(),
                link=(item.findtext("link") or "").strip() or None,
                description=_strip_html(desc),
                duration=(item.findtext(f"{ITUNES_NS}duration") or "").strip() or None,
            )
        )
    episodes.sort(key=lambda e: e.published, reverse=True)
    return episodes


def fetch_episodes(feed_url: str) -> list[Episode]:
    return parse_feed(http_get(feed_url))
