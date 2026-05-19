"""
preprocess.py
=============
Preprocessing pipeline for the Yogera Luganda speech corpus.

Steps (as described in manuscript Sections 4.4 and 5.1):
  1. Load metadata.csv and verify all referenced audio files exist
  2. Convert each audio file to 16 kHz, mono, 16-bit PCM WAV using ffmpeg
  3. Validate clip duration is within allowed bounds (1–20 seconds)
  4. Trim leading/trailing silence using pywebrtcvad [Wiseman, 2016]
  5. Recompute duration in hours and update metadata
  6. Write cleaned audio files and updated metadata.csv

Input metadata.csv columns (actual schema):
  sentence, language, contributor_id, gender, age_group,
  voice_clip, duration (hours), Region

Usage:
    python preprocess.py \
        --audio_dir   /path/to/raw_audio \
        --metadata    /path/to/metadata.csv \
        --output_dir  /path/to/output/audio \
        --output_meta /path/to/output/metadata.csv

Requirements:
    pip install pandas librosa webrtcvad
    ffmpeg must be on your PATH
"""

import os
import wave
import shutil
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import librosa
import webrtcvad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants matching the corpus spec ───────────────────────────────────────
TARGET_SR       = 16000   # Hz
TARGET_CHANNELS = 1       # mono
TARGET_SAMPWIDTH = 2      # 16-bit
MIN_DURATION_S  = 1.0     # seconds
MAX_DURATION_S  = 20.0    # seconds (max in actual data)
VAD_MODE        = 3       # webrtcvad aggressiveness 0–3
VAD_FRAME_MS    = 30      # ms per VAD frame (10, 20, or 30 only)
VAD_PADDING_MS  = 300     # ms of silence kept around speech


# ── audio helpers ─────────────────────────────────────────────────────────────

def convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert any audio file to 16 kHz mono 16-bit WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-sample_fmt", "s16",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        log.warning("ffmpeg failed on %s: %s", src.name,
                    result.stderr.decode()[:200])
        return False
    return True


def get_duration_seconds(path: Path) -> float:
    """Return audio duration in seconds using librosa."""
    return librosa.get_duration(path=str(path))


def read_pcm(path: Path):
    """Read a 16 kHz mono 16-bit WAV as raw PCM bytes."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate()   == TARGET_SR,       "Sample rate mismatch"
        assert wf.getnchannels()   == TARGET_CHANNELS, "Channel count mismatch"
        assert wf.getsampwidth()   == TARGET_SAMPWIDTH,"Sample width mismatch"
        return wf.readframes(wf.getnframes())


def write_pcm(path: Path, pcm: bytes):
    """Write raw 16-bit mono PCM bytes as a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(TARGET_CHANNELS)
        wf.setsampwidth(TARGET_SAMPWIDTH)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm)


def vad_trim(pcm: bytes) -> bytes:
    """
    Remove non-speech from the start and end of a 16 kHz 16-bit mono PCM
    buffer using pywebrtcvad.

    Algorithm:
      - Split into VAD_FRAME_MS frames
      - Find first and last frame classified as speech
      - Keep from (first_speech - padding) to (last_speech + padding)
    """
    vad       = webrtcvad.Vad(VAD_MODE)
    frame_bytes = int(TARGET_SR * VAD_FRAME_MS / 1000) * TARGET_SAMPWIDTH
    pad_frames  = VAD_PADDING_MS // VAD_FRAME_MS

    frames = [pcm[i: i + frame_bytes]
              for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes)]
    if not frames:
        return pcm

    speech = []
    for frame in frames:
        try:
            speech.append(vad.is_speech(frame, TARGET_SR))
        except Exception:
            speech.append(False)

    speech_idx = [i for i, s in enumerate(speech) if s]
    if not speech_idx:
        log.debug("VAD found no speech — returning original buffer")
        return pcm

    first = max(0, speech_idx[0] - pad_frames)
    last  = min(len(frames) - 1, speech_idx[-1] + pad_frames)
    return b"".join(frames[first: last + 1])


def duration_to_hours(seconds: float) -> float:
    """
    Round duration to the nearest whole second then convert to hours.
    Matches the rounding convention observed in the actual metadata.csv
    (durations stored as exact 1-second multiples in hours).
    """
    rounded_s = round(seconds)
    return round(rounded_s / 3600, 9)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(audio_dir: Path, metadata_path: Path,
        output_dir: Path, output_meta: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    # load metadata
    log.info("Loading metadata from %s", metadata_path)
    df = pd.read_csv(metadata_path, encoding="utf-8")

    expected_cols = {"sentence", "language", "contributor_id", "gender",
                     "age_group", "voice_clip", "duration", "Region"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    log.info("Loaded %d rows, %d unique speakers, %d unique sentences",
             len(df),
             df["contributor_id"].nunique(),
             df["sentence"].nunique())

    kept, dropped = [], 0

    for idx, row in df.iterrows():
        src = audio_dir / row["voice_clip"]

        if not src.exists():
            log.warning("[%d] Missing file: %s", idx, src.name)
            dropped += 1
            continue

        # convert to standard format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        ok = convert_to_wav(src, tmp_path)
        if not ok:
            tmp_path.unlink(missing_ok=True)
            dropped += 1
            continue

        # check duration
        dur_s = get_duration_seconds(tmp_path)
        if not (MIN_DURATION_S <= dur_s <= MAX_DURATION_S):
            log.debug("[%d] Duration %.1fs out of range, dropping %s",
                      idx, dur_s, src.name)
            tmp_path.unlink(missing_ok=True)
            dropped += 1
            continue

        # VAD trim
        try:
            pcm = read_pcm(tmp_path)
            pcm = vad_trim(pcm)
            write_pcm(tmp_path, pcm)
            dur_s = get_duration_seconds(tmp_path)
            # drop if trimming reduced duration below minimum
            if dur_s < MIN_DURATION_S:
                log.debug("[%d] Post-VAD duration too short, dropping %s",
                          idx, src.name)
                tmp_path.unlink(missing_ok=True)
                dropped += 1
                continue
        except Exception as e:
            log.warning("[%d] VAD failed on %s (%s), using untrimmed",
                        idx, src.name, e)

        # copy to output keeping original filename
        dst = output_dir / row["voice_clip"]
        shutil.move(str(tmp_path), str(dst))

        # update duration (rounded to nearest second, stored in hours)
        new_row = row.to_dict()
        new_row["duration"] = duration_to_hours(dur_s)
        kept.append(new_row)

        if (idx + 1) % 1000 == 0:
            log.info("Processed %d / %d clips ...", idx + 1, len(df))

    # write cleaned metadata preserving original column order
    col_order = ["sentence", "language", "contributor_id", "gender",
                 "age_group", "voice_clip", "duration", "Region"]
    out_df = pd.DataFrame(kept)[col_order]
    out_df.to_csv(output_meta, index=False, encoding="utf-8")

    log.info("Done. Kept=%d  Dropped=%d", len(kept), dropped)
    log.info("Output audio : %s", output_dir)
    log.info("Output meta  : %s", output_meta)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Yogera Luganda corpus audio files")
    parser.add_argument("--audio_dir",   required=True,
                        help="Directory containing raw WAV files")
    parser.add_argument("--metadata",    required=True,
                        help="Path to metadata.csv")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory for cleaned WAV files")
    parser.add_argument("--output_meta", required=True,
                        help="Path for cleaned metadata.csv")
    args = parser.parse_args()

    run(
        audio_dir     = Path(args.audio_dir),
        metadata_path = Path(args.metadata),
        output_dir    = Path(args.output_dir),
        output_meta   = Path(args.output_meta),
    )
