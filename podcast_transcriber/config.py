"""Configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Apple Podcasts ID for "All-In with Chamath, Jason, Sacks & Friedberg".
ALL_IN_APPLE_ID = "1502871393"

DEFAULT_HOSTS = [
    "Jason Calacanis",
    "Chamath Palihapitiya",
    "David Sacks",
    "David Friedberg",
]


@dataclass
class Config:
    data_dir: Path
    feed_url: str | None
    apple_podcast_id: str
    hosts: list[str] = field(default_factory=lambda: list(DEFAULT_HOSTS))

    # Transcription
    whisper_model: str = "large-v3-turbo"
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 8
    threads: int = 4
    hf_token: str | None = None
    min_speakers: int = 2
    max_speakers: int = 8

    # Speaker mapping
    anthropic_api_key: str | None = None
    claude_model: str = "claude-haiku-5-5"
    summary_model: str = "claude-opus-5-5"  # key takeaways; quality matters more than for naming

    # Email: Resend (preferred) or SMTP
    resend_api_key: str | None = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    email_from: str | None = None
    email_to: str | None = None  # recipients, comma-separated (the EMAIL_TO secret on Actions)
    recipients_file: Path = Path("recipients.txt")

    # Run behaviour
    max_per_run: int = 1
    initial_backlog: int = 1
    keep_audio: bool = False

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def episodes_dir(self) -> Path:
        return self.data_dir / "episodes"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        _load_dotenv(env_file or Path.cwd() / ".env")
        hosts = _env("HOSTS")
        smtp_user = _env("SMTP_USER")
        return cls(
            data_dir=Path(_env("DATA_DIR", "data")).expanduser().resolve(),
            feed_url=_env("FEED_URL"),
            apple_podcast_id=_env("APPLE_PODCAST_ID", ALL_IN_APPLE_ID),
            hosts=[h.strip() for h in hosts.split(",")] if hosts else list(DEFAULT_HOSTS),
            whisper_model=_env("WHISPER_MODEL", "large-v3-turbo"),
            device=_env("DEVICE", "cpu"),
            compute_type=_env("COMPUTE_TYPE", "int8"),
            batch_size=_env_int("BATCH_SIZE", 8),
            threads=_env_int("THREADS", os.cpu_count() or 4),
            hf_token=_env("HF_TOKEN"),
            min_speakers=_env_int("MIN_SPEAKERS", 2),
            max_speakers=_env_int("MAX_SPEAKERS", 8),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            claude_model=_env("CLAUDE_MODEL", "claude-haiku-5-5"),
            summary_model=_env("SUMMARY_MODEL", "claude-opus-5-5"),
            resend_api_key=_env("RESEND_API_KEY"),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=_env_int("SMTP_PORT", 587),
            smtp_user=smtp_user,
            smtp_password=_env("SMTP_PASSWORD"),
            email_from=_env("EMAIL_FROM"),
            email_to=_env("EMAIL_TO"),
            recipients_file=Path(_env("RECIPIENTS_FILE", "recipients.txt")).expanduser().resolve(),
            max_per_run=_env_int("MAX_PER_RUN", 1),
            initial_backlog=_env_int("INITIAL_BACKLOG", 1),
            keep_audio=_env_bool("KEEP_AUDIO", False),
        )

    def missing_for_run(self) -> list[str]:
        missing = []
        if not self.hf_token:
            missing.append("HF_TOKEN")
        try:
            if not self.recipients():
                missing.append("recipients (set EMAIL_TO)")
        except ValueError as exc:
            missing.append(f"valid recipients ({exc})")
        if not (self.resend_api_key or (self.smtp_user and self.smtp_password)):
            missing.append("RESEND_API_KEY (or SMTP_USER + SMTP_PASSWORD)")
        return missing

    def recipients(self) -> list[str]:
        from .recipients import load_recipients

        return load_recipients(self.recipients_file, self.email_to)

    @property
    def email_method(self) -> str:
        return "resend" if self.resend_api_key else "smtp"
