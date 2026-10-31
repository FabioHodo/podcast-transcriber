"""CLI: python -m podcast_transcriber [run|list|check] [options]"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .feed import fetch_episodes
from .pipeline import get_feed_url, pending_episodes, run
from .state import State


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="podcast_transcriber", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="process new episodes (default)")
    p_run.add_argument("--dry-run", action="store_true", help="transcribe but don't email or mark done")
    p_run.add_argument("--episode", help="'latest', a GUID, or part of a title")
    p_run.add_argument("--force", action="store_true", help="redo an episode even if already emailed")

    p_list = sub.add_parser("list", help="show recent feed episodes and their status")
    p_list.add_argument("-n", type=int, default=10)

    sub.add_parser("check", help="validate config and feed access without transcribing")
    p_test = sub.add_parser("test-email", help="send a small test email with an attachment")
    p_test.add_argument("--all", action="store_true",
                        help="send to every recipient (default: only the first one)")
    sub.add_parser("validate", help="check the recipient list (EMAIL_TO and recipients.txt)")
    sub.add_parser("pending", help="quick check: is there a new episode? (no ML deps needed)")

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config.from_env()
    cmd = args.cmd or "run"

    if cmd == "list":
        state = State(cfg.state_path)
        for ep in fetch_episodes(get_feed_url(cfg, state))[: args.n]:
            mark = "done" if state.is_done(ep.guid) else "new "
            print(f"[{mark}] {ep.date}  {ep.title}")
        return 0

    if cmd == "pending":
        import os

        eps = pending_episodes(cfg)
        for ep in eps:
            print(f"new: {ep.date}  {ep.title}")
        if not eps:
            print("Nothing new.")
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"new={'true' if eps else 'false'}\n")
        return 0

    if cmd == "validate":
        try:
            recipients = cfg.recipients()
        except ValueError as exc:
            print(f"Invalid recipient list:\n{exc}")
            return 1
        if not recipients:
            print("No recipients: set EMAIL_TO")
            return 1
        print(f"OK: {len(recipients)} recipient(s)")
        return 0

    if cmd == "test-email":
        import tempfile
        from pathlib import Path
        from .emailer import send_email
        from .recipients import mask

        recipients = cfg.recipients()
        if not recipients:
            print(f"No recipients: set EMAIL_TO (or list addresses in {cfg.recipients_file.name} locally)")
            return 1
        targets = recipients if args.all else recipients[:1]
        with tempfile.TemporaryDirectory() as tmp:
            att = Path(tmp) / "test-transcript.txt"
            att.write_text("[00:00:00] Test Speaker: If you can read this, attachments work.\n", encoding="utf-8")
            for addr in targets:
                send_email(cfg, addr, "Podcast transcriber: test email",
                           "Email delivery works. The real transcripts will arrive like this.\n",
                           "<p>Email delivery works. The real transcripts will arrive like this.</p>", att)
                print(f"Test email sent to {mask(addr)} via {cfg.email_method}.")
        return 0

    if cmd == "check":
        state = State(cfg.state_path)
        try:
            url = get_feed_url(cfg, state)
            eps = fetch_episodes(url)
        except Exception as exc:
            print(f"Feed check failed: {exc}")
            return 1
        print(f"Feed OK: {url}\nNewest: {eps[0].date}  {eps[0].title}\nAudio:  {eps[0].audio_url}")
        missing = cfg.missing_for_run()
        if not cfg.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY (optional: speakers stay unnamed)")
        print(f"Email method: {cfg.email_method}")
        try:
            print(f"Recipients: {len(cfg.recipients())}")
        except ValueError:
            pass
        print("Missing config: " + (", ".join(missing) if missing else "none"))
        return 0 if not cfg.missing_for_run() else 1

    do_email = not getattr(args, "dry_run", False)
    return run(cfg, send_email=do_email, only=getattr(args, "episode", None), force=getattr(args, "force", False))


if __name__ == "__main__":
    sys.exit(main())
