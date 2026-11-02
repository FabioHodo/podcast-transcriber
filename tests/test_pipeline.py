"""Offline tests: no network, no models. Run with `python -m pytest`."""

import json
from pathlib import Path

import pytest

from types import SimpleNamespace

from podcast_transcriber import pipeline, render, speakers, summary
from podcast_transcriber.config import Config
from podcast_transcriber import emailer
from podcast_transcriber.emailer import build_message
from podcast_transcriber.feed import parse_feed
from podcast_transcriber.recipients import RecipientsError, mask, parse_recipients
from podcast_transcriber.state import State
from podcast_transcriber.turns import Turn, apply_names, build_turns, format_ts, speaker_stats

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel><title>All-In</title>
<item><title>Older episode</title><guid>g1</guid>
  <pubDate>Fri, 16 Oct 2026 22:00:00 +0000</pubDate>
  <enclosure url="https://example.com/1.mp3" type="audio/mpeg"/>
  <itunes:duration>01:40:00</itunes:duration>
  <description><![CDATA[<p>Topics &amp; more</p>]]></description></item>
<item><title>AI Cyber Panic, Ken Griffin's Bubble Warning</title><guid>g2</guid>
  <pubDate>Fri, 23 Oct 2026 22:00:00 +0000</pubDate>
  <enclosure url="https://example.com/2.mp3" type="audio/mpeg"/>
  <description>With Nikesh Arora</description></item>
<item><title>No audio</title><guid>g3</guid></item>
</channel></rss>"""


def make_cfg(tmp_path: Path, **kw) -> Config:
    base = dict(data_dir=tmp_path, feed_url="https://example.com/feed", apple_podcast_id="1",
                smtp_user="me@example.com", smtp_password="x", email_from="me@example.com",
                email_to=None, recipients_file=tmp_path / "recipients.txt")
    base.update(kw)
    return Config(**base)


def word(w, s, e, spk=None):
    d = {"word": w, "start": s, "end": e}
    if spk:
        d["speaker"] = spk
    return d


SEGMENTS = [
    {"text": "Welcome back Chamath.", "words": [
        word("Welcome", 0.0, 0.4, "SPEAKER_00"), word("back", 0.4, 0.7, "SPEAKER_00"),
        word("Chamath.", 0.7, 1.2, "SPEAKER_00")]},
    {"text": "Thanks Jason, great to be here in 2026.", "words": [
        word("Thanks", 1.5, 1.8, "SPEAKER_01"), word("Jason,", 1.8, 2.1, "SPEAKER_01"),
        word("great", 2.2, 2.4, "SPEAKER_01"), word("yeah", 2.4, 2.5, "SPEAKER_00"),
        word("to", 2.5, 2.6, "SPEAKER_01"), word("be", 2.6, 2.7, "SPEAKER_01"),
        word("here", 2.7, 2.9, "SPEAKER_01"), word("in", 2.9, 3.0, "SPEAKER_01"),
        {"word": "2026."}]},  # no timing/speaker, like WhisperX numbers
    {"text": "Let's get started.", "speaker": "SPEAKER_00", "start": 4.0, "end": 5.0},
]


def test_parse_feed_sorts_newest_first_and_skips_no_audio():
    eps = parse_feed(FEED)
    assert [e.guid for e in eps] == ["g2", "g1"]
    assert eps[1].description == "Topics & more"
    assert eps[0].date == "2026-10-23"
    assert eps[0].slug.startswith("2026-10-23-ai-cyber-panic-ken-griffin")


def test_build_turns_inherits_and_smooths():
    turns = build_turns(SEGMENTS)
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert turns[1].text == "Thanks Jason, great yeah to be here in 2026."
    assert turns[2].text == "Let's get started."
    assert turns[2].start == 4.0


def test_apply_names_merges_split_labels_and_falls_back():
    turns = [Turn("SPEAKER_00", 0, 1, "a"), Turn("SPEAKER_02", 1, 2, "b"),
             Turn("SPEAKER_01", 2, 3, "c"), Turn("SPEAKER_03", 3, 4, "d")]
    named = apply_names(turns, {"SPEAKER_00": "Jason", "SPEAKER_02": "Jason", "SPEAKER_01": "Chamath"})
    assert [(t.speaker, t.text) for t in named] == [("Jason", "a b"), ("Chamath", "c"), ("Speaker 1", "d")]


def test_parse_mapping_tolerates_fences_and_drops_unknowns():
    reply = 'Sure:\n```json\n{"mapping": {"SPEAKER_00": "Jason Calacanis", "SPEAKER_01": "Unknown", "SPEAKER_09": "X"}, "notes": "ok"}\n```'
    mapping, data = speakers.parse_mapping(reply, {"SPEAKER_00", "SPEAKER_01"})
    assert mapping == {"SPEAKER_00": "Jason Calacanis"}
    assert data["notes"] == "ok"


def test_prompt_contains_evidence():
    ep = parse_feed(FEED)[0]
    prompt = speakers.build_user_prompt(ep, build_turns(SEGMENTS), ["Jason Calacanis", "Chamath Palihapitiya"])
    assert "Nikesh Arora" in prompt and "SPEAKER_01" in prompt and "[00:00:01] SPEAKER_01: Thanks" in prompt


def test_render_and_email(tmp_path):
    ep = parse_feed(FEED)[0]
    named = apply_names(build_turns(SEGMENTS), {"SPEAKER_00": "Jason Calacanis", "SPEAKER_01": "Chamath Palihapitiya"})
    text = render.transcript_text(ep, named)
    assert "[00:00:00] Jason Calacanis: Welcome back Chamath." in text
    plain, html = render.email_bodies(ep, named, {"notes": "high confidence"})
    assert "Chamath Palihapitiya" in plain and "<table" in html and "high confidence" in plain
    att = tmp_path / "t.txt"
    att.write_text(text, encoding="utf-8")
    msg = build_message(make_cfg(tmp_path), "me@example.com", render.email_subject(ep), plain, html, att)
    parts = [p.get_filename() for p in msg.iter_attachments()]
    assert parts == ["t.txt"] and msg["Subject"].startswith("All-In transcript:")


def test_format_ts_and_stats():
    assert format_ts(3725.9) == "01:02:05"
    stats = speaker_stats([Turn("A", 0, 10, "one two"), Turn("B", 10, 11, "x"), Turn("A", 11, 12, "y")])
    assert list(stats) == ["A", "B"] and stats["A"]["words"] == 3


def test_run_first_time_processes_only_latest_then_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "fetch_episodes", lambda url: parse_feed(FEED))
    processed = []

    def fake_process(ep, cfg, send_email=True):
        processed.append(ep.guid)
        return tmp_path / "x.txt"

    monkeypatch.setattr(pipeline, "process_episode", fake_process)
    cfg = make_cfg(tmp_path, email_to="me@example.com")
    assert pipeline.run(cfg) == 0
    assert processed == ["g2"]
    state = json.loads(cfg.state_path.read_text())
    assert set(state["done"]) == {"g1", "g2"}
    assert state["done"]["g1"]["note"].startswith("skipped")
    assert pipeline.run(cfg) == 0 and processed == ["g2"]


def test_run_failure_is_retried_next_time(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "fetch_episodes", lambda url: parse_feed(FEED))
    calls = {"n": 0}

    def flaky(ep, cfg, send_email=True):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("smtp down")
        return tmp_path / "x.txt"

    monkeypatch.setattr(pipeline, "process_episode", flaky)
    cfg = make_cfg(tmp_path, email_to="me@example.com")
    assert pipeline.run(cfg) == 1
    assert not State(cfg.state_path).is_done("g2")
    assert pipeline.run(cfg) == 0
    assert State(cfg.state_path).is_done("g2")


def test_process_episode_uses_cache_and_no_api_key(tmp_path, monkeypatch):
    """Full stage flow with a cached WhisperX result: no download, no model, no email."""
    ep = parse_feed(FEED)[0]
    cfg = make_cfg(tmp_path, anthropic_api_key=None)
    work = cfg.episodes_dir / ep.slug
    work.mkdir(parents=True)
    (work / "raw_whisperx.json").write_text(json.dumps({"segments": SEGMENTS}))
    out = pipeline.process_episode(ep, cfg, send_email=False)
    text = out.read_text(encoding="utf-8")
    assert "Speaker 1: Welcome back Chamath." in text
    assert json.loads((work / "speakers.json").read_text())["mapping"] == {}
    assert "KEY TAKEAWAYS" not in text
    assert not (work / "summary.json").exists()  # "no key" isn't cached; adding a key later works


TAKEAWAYS = [
    {"speaker": "Jason Calacanis", "timestamp": "00:00:00", "point": "Chamath is back on the show."},
    {"speaker": "Chamath Palihapitiya", "timestamp": "00:00:01", "point": "It is great to be here in 2026."},
]


def test_parse_takeaways_validates_and_caps():
    items = [{"speaker": "Jason Calacanis", "timestamp": "00:01:02", "point": "A real point."},
             {"speaker": "Made Up Person", "timestamp": "1:02", "point": "Unknown speaker, bad time."},
             {"speaker": "Jason Calacanis", "timestamp": "00:03:00", "point": "  "}]
    items += [{"speaker": "Jason Calacanis", "timestamp": "00:04:00", "point": f"p{i}"} for i in range(30)]
    out = summary.parse_takeaways(json.dumps({"takeaways": items}), {"Jason Calacanis"})
    assert out[0] == {"speaker": "Jason Calacanis", "timestamp": "00:01:02", "point": "A real point."}
    assert out[1] == {"speaker": "", "timestamp": "", "point": "Unknown speaker, bad time."}
    assert len(out) == summary.MAX_TAKEAWAYS and all(t["point"].strip() for t in out)


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned reply."""

    def __init__(self, text="", stop_reason="end_turn"):
        self.kwargs = None
        reply = SimpleNamespace(
            stop_reason=stop_reason, stop_details=None, model="claude-opus-5-5",
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        )
        create = lambda **kw: (setattr(self, "kwargs", kw), reply)[1]  # noqa: E731
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_summarize_request_and_reply(tmp_path):
    ep = parse_feed(FEED)[0]
    named = apply_names(build_turns(SEGMENTS), {"SPEAKER_00": "Jason Calacanis", "SPEAKER_01": "Chamath Palihapitiya"})
    client = FakeClient(json.dumps({"takeaways": TAKEAWAYS}))
    info = summary.summarize(ep, named, make_cfg(tmp_path, anthropic_api_key="k"), client=client)
    assert info == {"takeaways": TAKEAWAYS, "model": "claude-opus-5-5"}
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert "[00:00:01] Chamath Palihapitiya: Thanks" in kw["messages"][0]["content"]
    with pytest.raises(RuntimeError, match="declined"):
        summary.summarize(ep, named, make_cfg(tmp_path, anthropic_api_key="k"), client=FakeClient(stop_reason="refusal"))


def test_render_includes_takeaways(tmp_path):
    ep = parse_feed(FEED)[0]
    named = apply_names(build_turns(SEGMENTS), {"SPEAKER_00": "Jason Calacanis", "SPEAKER_01": "Chamath Palihapitiya"})
    text = render.transcript_text(ep, named, TAKEAWAYS)
    assert "- [00:00:01] Chamath Palihapitiya: It is great to be here in 2026." in text
    assert text.index("KEY TAKEAWAYS") < text.index("[00:00:00] Jason Calacanis: Welcome back")
    plain, html = render.email_bodies(ep, named, {}, TAKEAWAYS)
    assert "  - [00:00:00] Jason Calacanis: Chamath is back on the show." in plain
    assert plain.index("Key takeaways:") < plain.index("Who talked:")
    assert "<strong>Chamath Palihapitiya:</strong> It is great to be here in 2026.</li>" in html


def test_takeaways_failure_still_sends_transcript_and_cache_is_used(tmp_path, monkeypatch):
    ep = parse_feed(FEED)[0]
    cfg = make_cfg(tmp_path, anthropic_api_key=None)
    work = cfg.episodes_dir / ep.slug
    work.mkdir(parents=True)
    (work / "raw_whisperx.json").write_text(json.dumps({"segments": SEGMENTS}))
    sent = []
    monkeypatch.setattr(pipeline, "deliver", lambda c, w, subj, plain, html, att: sent.append(plain))

    def boom(*a, **k):
        raise RuntimeError("API down")

    monkeypatch.setattr(summary, "summarize", boom)
    pipeline.process_episode(ep, cfg, send_email=True)
    assert len(sent) == 1 and "Key takeaways" not in sent[0]

    (work / "summary.json").write_text(json.dumps({"takeaways": TAKEAWAYS, "model": "m"}))
    pipeline.process_episode(ep, cfg, send_email=True)  # cached: summarize (still broken) isn't called
    assert "Chamath is back on the show." in sent[1]


def test_resend_payload_and_dispatch(tmp_path, monkeypatch):
    import base64
    cfg = make_cfg(tmp_path, resend_api_key="re_test", email_from=None, email_to="me@example.com")
    att = tmp_path / "ep.txt"
    att.write_text("hello", encoding="utf-8")
    sent = {}
    monkeypatch.setattr(emailer, "send_resend", lambda c, payload: sent.update(payload))
    monkeypatch.setattr(emailer, "send_smtp", lambda c, m: pytest.fail("should use Resend"))
    emailer.send_email(cfg, "a@x.com", "subj", "plain", "<p>h</p>", att)
    assert sent["from"] == emailer.RESEND_DEFAULT_FROM
    assert sent["to"] == ["a@x.com"]
    assert base64.b64decode(sent["attachments"][0]["content"]) == b"hello"
    assert cfg.email_method == "resend" and cfg.missing_for_run() == ["HF_TOKEN"]


def test_missing_config_reports_email(tmp_path):
    cfg = make_cfg(tmp_path, smtp_user=None, smtp_password=None, hf_token="h")
    assert cfg.missing_for_run() == [
        "recipients (set EMAIL_TO)",
        "RESEND_API_KEY (or SMTP_USER + SMTP_PASSWORD)",
    ]


def test_parse_recipients_comments_dedupe_and_errors():
    text = """# header comment
    jane@example.com   # Jane
    Bob.Smith+news@mail.example.co.uk

    JANE@example.com
    """
    assert parse_recipients(text) == ["jane@example.com", "Bob.Smith+news@mail.example.co.uk"]
    with pytest.raises(RecipientsError) as exc:
        parse_recipients("ok@example.com\nnot-an-email\nalso@bad,com\n")
    assert "recipients.txt:2" in str(exc.value) and "recipients.txt:3" in str(exc.value)


def test_mask_hides_name_and_domain():
    assert mask("jane@example.com") == "j***@e***.com"
    assert mask("bob@mail.example.co.uk") == "b***@m***.uk"
    assert mask("x@localhost") == "x***@l***"


def test_public_repo_keeps_private_data_out():
    """The repo is public: no committed mailing list, no downloadable transcripts,
    and no workflow values that would be printed unmasked in public logs."""
    root = Path(__file__).resolve().parents[1]
    assert "recipients.txt" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    wf = (root / ".github/workflows/transcribe.yml").read_text(encoding="utf-8")
    assert "upload-artifact" not in wf
    assert "EMAIL_FROM: ${{ secrets.EMAIL_FROM }}" in wf and "vars.EMAIL_FROM" not in wf


def test_file_and_email_to_are_merged(tmp_path):
    (tmp_path / "recipients.txt").write_text("a@x.com\nb@x.com\n")
    cfg = make_cfg(tmp_path, email_to="private@x.com, a@x.com")
    assert cfg.recipients() == ["a@x.com", "b@x.com", "private@x.com"]


def test_bad_email_to_is_reported_as_email_to(tmp_path):
    """EMAIL_TO is the main list now; a typo must point at the secret, not recipients.txt."""
    cfg = make_cfg(tmp_path, email_to="ok@x.com, oops-x.com")
    with pytest.raises(RecipientsError, match=r"EMAIL_TO:2"):
        cfg.recipients()


def test_deliver_sends_one_copy_each_and_resumes_without_duplicates(tmp_path, monkeypatch):
    (tmp_path / "recipients.txt").write_text("a@x.com\nb@x.com\nc@x.com\n")
    cfg = make_cfg(tmp_path)
    work = tmp_path / "ep"
    work.mkdir()
    att = work / "t.txt"
    att.write_text("hi")
    calls = []
    fail = {"b@x.com"}

    def fake_send(c, to, *args):
        calls.append(to)
        if to in fail:
            raise RuntimeError("boom")

    monkeypatch.setattr(emailer, "send_email", fake_send)
    with pytest.raises(RuntimeError, match="1 recipient"):
        pipeline.deliver(cfg, work, "s", "p", "h", att)
    assert calls == ["a@x.com", "b@x.com", "c@x.com"]

    fail.clear()
    calls.clear()
    pipeline.deliver(cfg, work, "s", "p", "h", att)
    assert calls == ["b@x.com"]  # only the one that failed is retried


def test_run_stops_before_transcribing_on_bad_recipients(tmp_path, monkeypatch):
    (tmp_path / "recipients.txt").write_text("oops-no-at-sign\n")
    monkeypatch.setattr(pipeline, "fetch_episodes", lambda url: pytest.fail("should not reach feed"))
    assert pipeline.run(make_cfg(tmp_path)) == 1


def test_pending_is_read_only_and_matches_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "fetch_episodes", lambda url: parse_feed(FEED))
    cfg = make_cfg(tmp_path, email_to="me@example.com")
    assert [e.guid for e in pipeline.pending_episodes(cfg)] == ["g2"]  # first run: newest only
    assert not cfg.state_path.exists()                                  # nothing written
    monkeypatch.setattr(pipeline, "process_episode", lambda ep, c, send_email=True: None)
    assert pipeline.run(cfg) == 0
    assert pipeline.pending_episodes(cfg) == []


def test_pending_writes_github_output(tmp_path, monkeypatch, capsys):
    from podcast_transcriber import __main__ as cli
    monkeypatch.setattr(pipeline, "fetch_episodes", lambda url: parse_feed(FEED))
    monkeypatch.setattr(cli, "pending_episodes", pipeline.pending_episodes)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FEED_URL", "https://example.com/feed")
    out = tmp_path / "gh_out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert cli.main(["pending"]) == 0
    assert out.read_text() == "new=true\n"
    assert "AI Cyber Panic" in capsys.readouterr().out


def test_workflow_gates_heavy_install_on_pending_check():
    """The hourly schedule is only affordable if no-op runs skip the ML install."""
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/transcribe.yml").read_text(encoding="utf-8")
    assert 'cron: "17 * * * *"' in wf
    assert "python -m podcast_transcriber pending" in wf
    gate = "steps.pending.outputs.new == 'true'"
    for step in ("Install ffmpeg", "Install Python deps", "Cache models", "Run pipeline"):
        block = wf.split(f"- name: {step}", 1)[1].split("- name:", 1)[0]
        assert gate in block, f"{step} is not gated on the pending check"


def _step(wf: str, name: str) -> str:
    return wf.split(f"- name: {name}", 1)[1].split("- name:", 1)[0]


def test_workflow_auto_mode_behaves_like_the_schedule():
    """The external hourly trigger dispatches mode=auto: cheap check first, transcribe only
    when there's a new episode, and the pipeline itself runs as plain `run`."""
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/transcribe.yml").read_text(encoding="utf-8")
    assert "options: [run, auto, test-email, check]" in wf
    assert "inputs.mode == 'auto'" in _step(wf, "Check for new episodes")
    run_step = _step(wf, "Run pipeline")
    assert "(inputs.mode && inputs.mode != 'auto') || steps.pending.outputs.new == 'true'" in run_step
    assert "MODE: ${{ (!inputs.mode || inputs.mode == 'auto') && 'run' || inputs.mode }}" in run_step


def test_workflow_checks_out_current_branch_not_trigger_commit():
    """A run queued behind a long transcription must see the latest state.json, or it would
    transcribe and email the same episode a second time."""
    wf = (Path(__file__).resolve().parents[1] / ".github/workflows/transcribe.yml").read_text(encoding="utf-8")
    checkout = wf.split("uses: actions/checkout@v4", 1)[1].split("- uses:", 1)[0]
    assert "ref: ${{ github.event.repository.default_branch }}" in checkout
