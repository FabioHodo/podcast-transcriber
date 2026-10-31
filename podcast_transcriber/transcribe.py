"""Download an episode's audio and transcribe it with WhisperX (ASR + alignment + diarization).

Heavy imports (torch, whisperx) happen inside functions so the rest of the
package (feed checks, tests, email) works without them installed.
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import time
import urllib.request
from pathlib import Path

from .config import Config
from .feed import USER_AGENT

log = logging.getLogger(__name__)


def download_audio(url: str, dest: Path, retries: int = 3) -> Path:
    """Stream the MP3 to disk (resumes nothing; just retries on failure)."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Audio already downloaded: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            tmp.replace(dest)
            log.info("Downloaded %.1f MB", dest.stat().st_size / 1e6)
            return dest
        except Exception as exc:  # network hiccups are common on long downloads
            log.warning("Download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    return dest


def transcribe(audio_path: Path, cfg: Config) -> dict:
    """Return WhisperX's result dict: {"segments": [...]} with word-level speakers."""
    import whisperx  # noqa: WPS433 (heavy, optional import)
    from whisperx.diarize import DiarizationPipeline

    if not cfg.hf_token:
        raise RuntimeError(
            "HF_TOKEN is required for speaker diarization (pyannote). "
            "See README: create a Hugging Face token and accept the model terms."
        )

    t0 = time.time()
    audio = whisperx.load_audio(str(audio_path))
    minutes = len(audio) / 16000 / 60
    log.info("Loaded %.1f min of audio", minutes)

    # 1) Speech recognition
    model = whisperx.load_model(
        cfg.whisper_model,
        cfg.device,
        compute_type=cfg.compute_type,
        language="en",
        threads=cfg.threads,
    )
    result = model.transcribe(audio, batch_size=cfg.batch_size, language="en")
    log.info("ASR done in %.1f min (%d segments)", (time.time() - t0) / 60, len(result["segments"]))
    del model
    gc.collect()

    # 2) Word-level alignment (makes speaker assignment much more precise)
    t1 = time.time()
    align_model, metadata = whisperx.load_align_model(language_code="en", device=cfg.device)
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, cfg.device, return_char_alignments=False
    )
    log.info("Alignment done in %.1f min", (time.time() - t1) / 60)
    del align_model
    gc.collect()

    # 3) Diarization: who spoke when
    t2 = time.time()
    diarizer = DiarizationPipeline(token=cfg.hf_token, device=cfg.device)
    diarize_df = diarizer(audio, min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers)
    result = whisperx.assign_word_speakers(diarize_df, result)
    log.info("Diarization done in %.1f min", (time.time() - t2) / 60)
    del diarizer
    gc.collect()

    log.info("Transcription pipeline finished in %.1f min", (time.time() - t0) / 60)
    return {"segments": result["segments"], "audio_minutes": round(minutes, 1)}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy floats sneak into WhisperX output; coerce anything odd to float/str
    path.write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
