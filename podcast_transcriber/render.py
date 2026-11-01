"""Render the transcript attachment and the short email body."""

from __future__ import annotations

import html
import textwrap

from .feed import Episode
from .turns import Turn, format_ts, speaker_stats


def _fmt_minutes(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _takeaway_prefix(t: dict) -> str:
    """'[00:12:34] Chamath Palihapitiya: ' (either part may be missing)."""
    ts = f"[{t['timestamp']}] " if t.get("timestamp") else ""
    who = f"{t['speaker']}: " if t.get("speaker") else ""
    return ts + who


def transcript_text(episode: Episode, turns: list[Turn], takeaways: list[dict] | None = None,
                    width: int = 100) -> str:
    stats = speaker_stats(turns)
    total_words = sum(s["words"] for s in stats.values())
    header = [
        episode.title,
        "=" * min(len(episode.title), width),
        f"Published: {episode.date}",
    ]
    if episode.duration:
        header.append(f"Duration:  {episode.duration}")
    if episode.link:
        header.append(f"Link:      {episode.link}")
    header.append(f"Speakers:  {', '.join(stats)}")
    header.append(f"Words:     {total_words:,}")
    header.append("")
    header.append("Machine transcript (Whisper + speaker diarization). Names are inferred and may be wrong in places.")
    if takeaways:
        header += ["", "KEY TAKEAWAYS (AI summary)", ""]
        header += [
            textwrap.fill(t["point"], width=width, initial_indent="- " + _takeaway_prefix(t),
                          subsequent_indent="  ", break_long_words=False, break_on_hyphens=False)
            for t in takeaways
        ]
    header.append("-" * width)

    body = []
    for t in turns:
        prefix = f"[{format_ts(t.start)}] {t.speaker}: "
        body.append(
            textwrap.fill(
                t.text,
                width=width,
                initial_indent=prefix,
                subsequent_indent=" " * 11,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(header) + "\n\n" + "\n\n".join(body) + "\n"


def email_subject(episode: Episode) -> str:
    return f"All-In transcript: {episode.title}"


def email_bodies(episode: Episode, turns: list[Turn], speaker_info: dict,
                 takeaways: list[dict] | None = None) -> tuple[str, str]:
    """Return (plain_text, html) for a short email; the full transcript is attached."""
    takeaways = takeaways or []
    stats = speaker_stats(turns)
    total = sum(s["words"] for s in stats.values()) or 1
    runtime = turns[-1].end if turns else 0
    speaker_rows = [
        (name, s["words"] / total * 100, s["seconds"]) for name, s in stats.items()
    ]
    desc = episode.description.strip()
    if len(desc) > 900:
        desc = desc[:900].rsplit(" ", 1)[0] + " …"
    notes = speaker_info.get("notes") or ""

    plain = [
        episode.title,
        f"Published {episode.date} · runtime {_fmt_minutes(runtime)} · {total:,} words",
        "",
    ]
    if takeaways:
        plain += ["Key takeaways:", *[f"  - {_takeaway_prefix(t)}{t['point']}" for t in takeaways], ""]
    plain += [
        "Who talked:",
        *[f"  - {n}: {pct:.0f}% ({_fmt_minutes(sec)})" for n, pct, sec in speaker_rows],
    ]
    if notes:
        plain += ["", f"Speaker labelling note: {notes}"]
    plain += ["", "Episode description:", desc]
    if episode.link:
        plain += ["", episode.link]
    plain += ["", "Full speaker-labelled transcript attached."]

    rows = "".join(
        f"<tr><td style='padding:2px 12px 2px 0'>{html.escape(n)}</td>"
        f"<td style='padding:2px 12px 2px 0;text-align:right'>{pct:.0f}%</td>"
        f"<td style='color:#666'>{_fmt_minutes(sec)}</td></tr>"
        for n, pct, sec in speaker_rows
    )
    link = (
        f"<p><a href='{html.escape(episode.link)}'>Episode page</a></p>" if episode.link else ""
    )
    note_html = f"<p style='color:#666'><em>{html.escape(notes)}</em></p>" if notes else ""
    takeaways_html = ""
    if takeaways:
        items = "".join(
            "<li style='margin:0 0 6px'>"
            + (f"<span style='color:#666'>{html.escape(t['timestamp'])}</span> " if t.get("timestamp") else "")
            + (f"<strong>{html.escape(t['speaker'])}:</strong> " if t.get("speaker") else "")
            + f"{html.escape(t['point'])}</li>"
            for t in takeaways
        )
        takeaways_html = (
            "<p style=\"margin:0 0 4px\"><strong>Key takeaways</strong></p>"
            f"<ul style=\"margin:0 0 16px;padding-left:20px\">{items}</ul>"
        )
    html_body = f"""<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:15px;line-height:1.5;max-width:640px">
<h2 style="margin:0 0 4px">{html.escape(episode.title)}</h2>
<p style="color:#666;margin:0 0 16px">Published {episode.date} · runtime {_fmt_minutes(runtime)} · {total:,} words</p>
{takeaways_html}
<p style="margin:0 0 4px"><strong>Who talked</strong></p>
<table style="border-collapse:collapse;margin-bottom:12px">{rows}</table>
{note_html}
<p style="margin:16px 0 4px"><strong>Episode description</strong></p>
<p style="white-space:pre-wrap;margin:0">{html.escape(desc)}</p>
{link}
<p style="color:#666">Full speaker-labelled transcript attached.</p>
</div>"""
    return "\n".join(plain) + "\n", html_body
