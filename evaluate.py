"""
evaluate.py
===========
Evaluation and bias assessment for the Yogera Luganda ASR experiments.
Reproduces Tables 6–10 from the manuscript.

  Table 6  — WER for three fine-tuned models on the test set
  Table 7  — Cross-dataset WER (our data vs Common Voice Luganda)
  Table 8  — WER by gender (Male / Female)
  Table 9  — WER by gender and age group
  Table 10 — WER by Ugandan regional accent

Age groups in metadata.csv: 18-29, 30-39, 40-49, 50-59, 60-69, 70-79
For evaluation (Table 9), groups 50-59, 60-69, 70-79 are merged into >=50
as described in manuscript Section 5.1.

Regions in metadata.csv: Central, Eastern, Northern, Western

Usage:
    python evaluate.py \
        --audio_dir    /path/to/cleaned_audio \
        --metadata     /path/to/checkpoints/metadata_with_splits.csv \
        --model_dir    /path/to/checkpoints \
        --output_dir   /path/to/results \
        [--cv_audio_dir /path/to/common_voice_luganda] \
        [--cv_metadata  /path/to/cv_metadata.csv]

Requirements:
    pip install transformers datasets evaluate jiwer torch torchaudio
"""

import logging
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, Audio
import evaluate as hf_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

wer_metric = hf_evaluate.load("wer")

TARGET_SR = 16000

MODEL_PATHS = {
    "xlsr"        : "xlsr/best_model",
    "wav2vec2bert": "wav2vec2bert/best_model",
    "parakeet"    : "parakeet/best_model",
}

# age group merge map — matches manuscript Section 5.1
AGE_REMAP = {
    "18-29": "18-29",
    "30-39": "30-39",
    "40-49": "40-49",
    "50-59": ">=50",
    "60-69": ">=50",
    "70-79": ">=50",
}

REGIONS  = ["Central", "Eastern", "Northern", "Western"]
GENDERS  = ["Female", "Male"]
AGE_EVAL = ["18-29", "30-39", "40-49", ">=50"]


# ── dataset helpers ───────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, audio_dir: Path,
                  split: str = None) -> Dataset:
    """
    Build a HuggingFace Dataset from a metadata DataFrame.
    voice_clip filenames are like:
        yogera_text_audio_20240425_113518.961214_2374.wav
    """
    if split:
        df = df[df["split"] == split].reset_index(drop=True)

    data = {
        "path"          : [str(audio_dir / r) for r in df["voice_clip"]],
        "sentence"      : df["sentence"].tolist(),
        "gender"        : df["gender"].tolist(),
        "age_group"     : [AGE_REMAP.get(a, a) for a in df["age_group"]],
        "Region"        : df["Region"].tolist(),
        "contributor_id": df["contributor_id"].tolist(),
    }
    ds = Dataset.from_dict(data)
    ds = ds.cast_column("path", Audio(sampling_rate=TARGET_SR))
    return ds


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe(ds: Dataset, model_path: str,
               batch_size: int = 16) -> list:
    """Run ASR inference using a HuggingFace pipeline."""
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe   = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        chunk_length_s=30,
    )
    audio_arrays = [row["path"]["array"] for row in ds]
    predictions  = []

    for i in range(0, len(audio_arrays), batch_size):
        batch  = audio_arrays[i: i + batch_size]
        result = pipe(batch, batch_size=batch_size)
        predictions.extend([r["text"].lower().strip() for r in result])
        log.info("  Transcribed %d / %d",
                 min(i + batch_size, len(audio_arrays)), len(audio_arrays))

    return predictions


def transcribe_parakeet(ds: Dataset, model_path: str) -> list:
    """Run ASR inference using a NeMo Parakeet model."""
    try:
        import nemo.collections.asr as nemo_asr
        import soundfile as sf
        import tempfile, os
    except ImportError:
        log.error("Install NeMo: pip install nemo_toolkit['asr'] soundfile")
        return []

    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        str(Path(model_path) / "parakeet_luganda.nemo")
    )
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i, row in enumerate(ds):
            p = os.path.join(tmpdir, f"{i:05d}.wav")
            sf.write(p, row["path"]["array"], TARGET_SR)
            paths.append(p)
        results = model.transcribe(paths)
        preds   = results[0] if isinstance(results, tuple) else results

    return [p.lower().strip() for p in preds]


def wer_pct(preds: list, refs: list) -> float:
    """Return WER as percentage rounded to 1 decimal place."""
    score = wer_metric.compute(predictions=preds, references=refs)
    return round(score * 100, 1)


# ── table generators ──────────────────────────────────────────────────────────

def table6(test_ds: Dataset, model_dir: Path) -> pd.DataFrame:
    """
    Table 6 — WER for each of the three fine-tuned models on the test set.
    """
    refs = [s.lower().strip() for s in test_ds["sentence"]]
    rows = []
    for key, rel_path in MODEL_PATHS.items():
        full = str(model_dir / rel_path)
        log.info("[Table 6] Evaluating %s ...", key)
        preds = (transcribe_parakeet(test_ds, full)
                 if key == "parakeet"
                 else transcribe(test_ds, full))
        wer = wer_pct(preds, refs)
        rows.append({"Fine-tuned ASR Model": key, "WER (%)": wer})
        log.info("  WER = %.1f%%", wer)
    return pd.DataFrame(rows)


def table7(our_test: Dataset, cv_ds: Dataset, model_dir: Path) -> pd.DataFrame:
    """
    Table 7 — Cross-dataset WER.
    Compares fine-tuned XLS-R-300M against the pre-existing 53-language model.
    """
    best_path = str(model_dir / MODEL_PATHS["xlsr"])
    cv_path   = "facebook/wav2vec2-xls-r-300m"   # pre-existing baseline

    configs = [
        ("XLS-R wav2vec2-300M (fine-tuned)", best_path, our_test, "Our data",     "Our data"),
        ("XLS-R wav2vec2-300M (fine-tuned)", best_path, cv_ds,    "Our data",     "Common Voice"),
        ("XLS-R wav2vec2 (53 languages)",    cv_path,   our_test, "Common Voice", "Our data"),
        ("XLS-R wav2vec2 (53 languages)",    cv_path,   cv_ds,    "Common Voice", "Common Voice"),
    ]
    rows = []
    for model_name, path, ds, train_data, test_data in configs:
        log.info("[Table 7] %s | train=%s | test=%s", model_name, train_data, test_data)
        preds = transcribe(ds, path)
        refs  = [s.lower().strip() for s in ds["sentence"]]
        wer   = wer_pct(preds, refs)
        rows.append({"Model": model_name, "Training Data": train_data,
                     "Test Data": test_data, "WER (%)": wer})
    return pd.DataFrame(rows)


def table8(our_test: Dataset, cv_ds: Dataset, model_path: str) -> pd.DataFrame:
    """
    Table 8 — WER by gender on our data and Common Voice.
    Expected result from manuscript: Our data Male=5.3%, Female=5.9%
    """
    rows = []
    for ds, label in [(our_test, "Our data"), (cv_ds, "Common Voice")]:
        preds = transcribe(ds, model_path)
        refs  = [s.lower().strip() for s in ds["sentence"]]
        genders = ds["gender"]
        row = {"Dataset": label}
        for g in GENDERS:
            idx  = [i for i, x in enumerate(genders) if x == g]
            wer  = wer_pct([preds[i] for i in idx], [refs[i] for i in idx])
            row[f"{g} WER (%)"] = wer
            log.info("[Table 8] %s | %s WER=%.1f%%  (n=%d)", label, g, wer, len(idx))
        rows.append(row)
    return pd.DataFrame(rows)


def table9(test_ds: Dataset, model_path: str) -> pd.DataFrame:
    """
    Table 9 — WER by gender and (merged) age group.
    Age groups 50-59, 60-69, 70-79 are already merged to >=50 in build_dataset().
    Expected: Female 30-39 = 12.3% (highest), all others < 10%
    """
    preds   = transcribe(test_ds, model_path)
    refs    = [s.lower().strip() for s in test_ds["sentence"]]
    genders = test_ds["gender"]
    ages    = test_ds["age_group"]   # already remapped

    rows = []
    for g in GENDERS:
        for a in AGE_EVAL:
            idx = [i for i, (gi, ai) in enumerate(zip(genders, ages))
                   if gi == g and ai == a]
            if not idx:
                continue
            wer = wer_pct([preds[i] for i in idx], [refs[i] for i in idx])
            rows.append({"Gender": g, "Age Group": a, "WER (%)": wer})
            log.info("[Table 9] %s | %s WER=%.1f%%  (n=%d)", g, a, wer, len(idx))
    return pd.DataFrame(rows)


def table10(test_ds: Dataset, model_path: str) -> pd.DataFrame:
    """
    Table 10 — WER by Ugandan regional accent.
    Expected: Central=4.5%, Eastern=3.5%, Northern=8.1%, Western=6.3%
    Note: Northern has 88.4% of recordings from 18-29 age group,
          which likely explains higher WER.
    """
    preds   = transcribe(test_ds, model_path)
    refs    = [s.lower().strip() for s in test_ds["sentence"]]
    regions = test_ds["Region"]

    rows = []
    for r in REGIONS:
        idx = [i for i, x in enumerate(regions) if x == r]
        wer = wer_pct([preds[i] for i in idx], [refs[i] for i in idx])
        rows.append({"Regional Accent": f"{r} Uganda", "Average WER (%)": wer})
        log.info("[Table 10] %s WER=%.1f%%  (n=%d)", r, wer, len(idx))
    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Yogera Luganda ASR models")
    parser.add_argument("--audio_dir",    required=True)
    parser.add_argument("--metadata",     required=True,
                        help="metadata_with_splits.csv from train.py")
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--cv_audio_dir", default=None,
                        help="Common Voice Luganda audio directory (optional)")
    parser.add_argument("--cv_metadata",  default=None,
                        help="Common Voice metadata CSV (optional)")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    model_dir  = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading metadata ...")
    df      = pd.read_csv(args.metadata, encoding="utf-8")
    test_ds = build_dataset(df, audio_dir, split="test")
    log.info("Test set: %d clips  (%d Female, %d Male)",
             len(test_ds),
             df[df["split"] == "test"]["gender"].eq("Female").sum(),
             df[df["split"] == "test"]["gender"].eq("Male").sum())

    # Common Voice (optional)
    cv_ds = None
    if args.cv_audio_dir and args.cv_metadata:
        cv_df = pd.read_csv(args.cv_metadata, encoding="utf-8")
        cv_ds = build_dataset(cv_df, Path(args.cv_audio_dir))
        log.info("Common Voice: %d clips", len(cv_ds))

    best_model = str(model_dir / MODEL_PATHS["xlsr"])

    # Table 6 — all three models
    t6 = table6(test_ds, model_dir)
    t6.to_csv(output_dir / "table6_model_wer.csv", index=False)
    print("\nTable 6 — Model WER:\n", t6.to_string(index=False))

    if cv_ds:
        # Table 7 — cross-dataset
        t7 = table7(test_ds, cv_ds, model_dir)
        t7.to_csv(output_dir / "table7_cross_dataset_wer.csv", index=False)
        print("\nTable 7 — Cross-dataset WER:\n", t7.to_string(index=False))

        # Table 8 — gender
        t8 = table8(test_ds, cv_ds, best_model)
        t8.to_csv(output_dir / "table8_gender_wer.csv", index=False)
        print("\nTable 8 — Gender WER:\n", t8.to_string(index=False))
    else:
        log.info("Skipping Tables 7 and 8 (no Common Voice data provided)")

    # Table 9 — age group x gender
    t9 = table9(test_ds, best_model)
    t9.to_csv(output_dir / "table9_age_gender_wer.csv", index=False)
    print("\nTable 9 — Age x Gender WER:\n", t9.to_string(index=False))

    # Table 10 — regional accent
    t10 = table10(test_ds, best_model)
    t10.to_csv(output_dir / "table10_regional_wer.csv", index=False)
    print("\nTable 10 — Regional WER:\n", t10.to_string(index=False))

    log.info("All results saved to %s", output_dir)
