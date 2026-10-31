# podcast-transcriber

Transcribes new podcast episodes with speaker names and emails the transcript.

```
RSS feed ──► new episode? ──► WhisperX (speech-to-text + speaker diarization)
         ──► Claude maps SPEAKER_00… to real names ──► email with the transcript attached
```

The transcript looks like:

```
[00:08:43] Jane Doe: I'm doing great. Thank you for having me.

[00:08:46] John Smith: And you had a nice event last week, didn't you?
```

## Setup

It needs a few credentials: access to the speaker-diarization model, an email provider, and
optionally an Anthropic API key for naming the speakers (without it they are "Speaker 1",
"Speaker 2", …). It also needs the list of people to email. Provide these as secrets when it runs
as a scheduled workflow, or in a local `.env` file; never put them in a committed file.
`podcast_transcriber/config.py` lists every setting and its default.

## Running locally

Needs Python 3.10–3.13 and `ffmpeg` on PATH.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m podcast_transcriber check                  # validate feed access + config
python -m podcast_transcriber test-email             # send a test email
python -m podcast_transcriber list                   # recent episodes and their status
python -m podcast_transcriber run --episode latest --dry-run   # transcribe, don't email
python -m podcast_transcriber run                    # new episodes -> email
```

Tests run offline: `python -m pytest`.

Transcripts are for personal use; don't republish them.
