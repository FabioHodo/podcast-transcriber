"""Load and validate the recipient list: the EMAIL_TO secret (comma-separated) and, for local
runs, an optional gitignored recipients.txt (one address per line)."""

from __future__ import annotations

import re
from pathlib import Path

# Deliberately simple: catches typos like missing "@" or a stray comma, not full RFC 5322.
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")


class RecipientsError(ValueError):
    pass


def parse_recipients(text: str, source: str = "recipients.txt") -> list[str]:
    """Return unique addresses in file order. Blank lines and '#' comments are ignored,
    including trailing comments ("bob@x.com  # Bob"). Raises on any malformed line."""
    seen: set[str] = set()
    out: list[str] = []
    errors: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        addr = raw.split("#", 1)[0].strip()
        if not addr:
            continue
        if not EMAIL_RE.match(addr):
            errors.append(f"{source}:{lineno}: not a valid email address: {addr!r}")
            continue
        key = addr.lower()
        if key not in seen:
            seen.add(key)
            out.append(addr)
    if errors:
        raise RecipientsError("\n".join(errors))
    return out


def load_recipients(path: Path, extra: str | None = None) -> list[str]:
    """Addresses from `path` (if it exists) plus any comma-separated `extra` (EMAIL_TO).

    Errors name their real source ("EMAIL_TO:2" is the second address in the secret).
    """
    text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    from_file = parse_recipients(text, source=path.name)
    from_env = parse_recipients("\n".join(a.strip() for a in extra.split(",")), source="EMAIL_TO") if extra else []
    return parse_recipients("\n".join(from_file + from_env))  # de-duplicate across both


def mask(addr: str) -> str:
    """j***@e***.com, for logs (Actions logs are public on public repos).

    The domain is masked too: a personal domain identifies someone as well as a name does.
    """
    local, _, domain = addr.partition("@")
    host, dot, tld = domain.rpartition(".")
    if not dot:
        host, tld = domain, ""
    return f"{local[:1]}***@{host[:1]}***{dot}{tld}"
