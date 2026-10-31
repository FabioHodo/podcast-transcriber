"""Orchestrate: feed -> new episodes -> audio -> transcript -> names -> email.

Every stage writes its output into data/episodes/<slug>/ and is skipped on rerun
if that output already exists, so a failed email never forces a re-transcription.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import emailer, render, speakers, transcribe
from .config import Config
from .feed import Episode, fetch_episodes, resolve_feed_url
from .recipients import mask
from .state import State
from .turns import Turn, apply_names, build_turns

log = logging.getLogger(__name__)


def get_feed_url(cfg: Config, state: State) -> str:
    if cfg.feed_url:
        return cfg.feed_url
    if state.feed_url:
        return state.feed_url
    url = resolve_feed_url(cfg.apple_podcast_id)
    log.info("Resolved feed URL via Apple lookup: %s", url)
    state.feed_url = url
    return url


def seed_backlog(state: State, episodes: list[Episode], keep_latest: int) -> None:
    """First run: don't transcribe the whole back catalogue, only the latest N."""
    for ep in episodes[keep_latest:]:
        state.mark_done(ep.guid, ep.title, note="skipped (backlog at first run)")
    log.info(
        "First run: marked %d older episodes as done; %d will be processed",
        max(0, len(episodes) - keep_latest), min(keep_latest, len(episodes)),
    )


def process_episode(ep: Episode, cfg: Config, send_email: bool = True) -> Path:
    """Run every stage for one episode; returns the transcript path."""
    work = cfg.episodes_dir / ep.slug
    work.mkdir(parents=True, exist_ok=True)
    (work / "episode.json").write_text(json.dumps(ep.to_dict(), indent=2))

    raw_path = work / "raw_whisperx.json"
    turns_path = work / "turns.json"
    names_path = work / "speakers.json"
    transcript_path = work / f"{ep.slug}.txt"
    audio_path = work / "audio.mp3"

    # 1-2) Audio + WhisperX
    if not raw_path.exists():
        log.info("Downloading audio for: %s", ep.title)
        transcribe.download_audio(ep.audio_url, audio_path)
        result = transcribe.transcribe(audio_path, cfg)
        transcribe.save_json(result, raw_path)
    else:
        log.info("Using cached WhisperX output")
        result = json.loads(raw_path.read_text())
    if audio_path.exists() and not cfg.keep_audio:
        audio_path.unlink()

    turns = build_turns(result["segments"])
    turns_path.write_text(json.dumps([t.to_dict() for t in turns], indent=1))

    # 3) Speaker names
    if names_path.exists():
        speaker_info = json.loads(names_path.read_text())
    else:
        speaker_info = speakers.map_speakers(ep, turns, cfg)
        names_path.write_text(json.dumps(speaker_info, indent=2))
    named: list[Turn] = apply_names(turns, speaker_info.get("mapping") or {})

    # 4) Render
    transcript_path.write_text(render.transcript_text(ep, named), encoding="utf-8")
    log.info("Transcript written: %s (%d turns)", transcript_path, len(named))

    # 5) Email
    if send_email:
        plain, html = render.email_bodies(ep, named, speaker_info)
        deliver(cfg, work, render.email_subject(ep), plain, html, transcript_path)
    return transcript_path


def deliver(cfg: Config, work: Path, subject: str, plain: str, html: str, attachment: Path) -> None:
    """Send one copy per recipient, remembering who already got it.

    If some sends fail, the rerun only retries those people, so nobody gets duplicates.
    """
    sent_path = work / "sent.json"
    sent: list[str] = json.loads(sent_path.read_text()) if sent_path.exists() else []
    already = {a.lower() for a in sent}
    recipients = cfg.recipients()
    failed: list[str] = []
    for addr in recipients:
        if addr.lower() in already:
            continue
        try:
            emailer.send_email(cfg, addr, subject, plain, html, attachment)
        except Exception as exc:
            failed.append(addr)
            log.error("Send to %s failed: %s", mask(addr), exc)
            continue
        sent.append(addr)
        sent_path.write_text(json.dumps(sent, indent=1))
    got_it = {a.lower() for a in sent}
    delivered = sum(1 for a in recipients if a.lower() in got_it)
    log.info("Delivered to %d/%d recipients via %s", delivered, len(recipients), cfg.email_method)
    if failed:
        raise RuntimeError(f"Email failed for {len(failed)} recipient(s): {', '.join(map(mask, failed))}")


def pending_episodes(cfg: Config) -> list[Episode]:
    """Episodes the next scheduled run would process. Read-only: never writes state.

    Used by the workflow's cheap pre-check so the multi-GB ML install only happens
    when there is actually something to transcribe.
    """
    state = State(cfg.state_path)
    episodes = fetch_episodes(get_feed_url(cfg, state))
    if not state.exists:
        return episodes[: cfg.initial_backlog]
    return [e for e in episodes if not state.is_done(e.guid)]


def run(
    cfg: Config,
    send_email: bool = True,
    only: str | None = None,
    force: bool = False,
) -> int:
    """Process new episodes. Returns a process exit code (0 = all good)."""
    if send_email:
        # Fail fast on a bad recipient list, before spending hours transcribing.
        try:
            recipients = cfg.recipients()
        except ValueError as exc:
            log.error("Invalid recipient list:\n%s", exc)
            return 1
        if not recipients:
            log.error("No recipients: set EMAIL_TO (or list addresses in %s locally)", cfg.recipients_file.name)
            return 1
        log.info("%d recipient(s) configured", len(recipients))

    state = State(cfg.state_path)
    feed_url = get_feed_url(cfg, state)
    episodes = fetch_episodes(feed_url)
    if not episodes:
        log.error("Feed has no episodes with audio: %s", feed_url)
        return 1
    log.info("Feed has %d episodes; newest: %s (%s)", len(episodes), episodes[0].title, episodes[0].date)

    if only:
        if only == "latest":
            targets = episodes[:1]
        else:
            q = only.lower()
            targets = [e for e in episodes if e.guid == only or q in e.title.lower()][:1]
        if not targets:
            log.error("No episode matches %r", only)
            return 1
        if not force:
            targets = [e for e in targets if not state.is_done(e.guid)]
            if not targets:
                log.info("Already done (use --force to redo): %s", only)
                state.save()
                return 0
    else:
        if not state.exists:
            seed_backlog(state, episodes, cfg.initial_backlog)
        pending = [e for e in episodes if not state.is_done(e.guid)]
        targets = list(reversed(pending[: cfg.max_per_run]))  # oldest of the batch first
    state.save()

    if not targets:
        log.info("Nothing new.")
        return 0

    failures = 0
    for ep in targets:
        log.info("=== %s (%s) ===", ep.title, ep.date)
        try:
            process_episode(ep, cfg, send_email=send_email)
            if send_email:
                state.mark_done(ep.guid, ep.title)
                state.save()
        except Exception:
            failures += 1
            log.exception("Failed on episode: %s", ep.title)
    return 1 if failures else 0
