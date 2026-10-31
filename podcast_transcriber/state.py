"""Tiny JSON state file: which episode GUIDs have already been emailed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class State:
    def __init__(self, path: Path):
        self.path = path
        self.exists = path.exists()
        data = json.loads(path.read_text()) if self.exists else {}
        self.done: dict[str, dict] = data.get("done", {})
        self.feed_url: str | None = data.get("feed_url")

    def is_done(self, guid: str) -> bool:
        return guid in self.done

    def mark_done(self, guid: str, title: str, note: str = "emailed") -> None:
        self.done[guid] = {
            "title": title,
            "note": note,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"feed_url": self.feed_url, "done": self.done}, indent=2))
        tmp.replace(self.path)
        self.exists = True
