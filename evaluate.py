"""
evaluate.py
===========
Evaluation and bias assessment for the Yogera Luganda ASR experiments.
Reproduces Tables 6-10 from the manuscript.

  Table 6  — WER for three fine-tuned models on the test set
  Table 7  — Cross-dataset WER (our data vs Common Voice Luganda)
  Table 8  — WER by gender (Male / Female)
  Table 9  — WER by gender and age group
  Table 10 — WER by Ugandan regional accent

Fine-tuned models used:
  XLS-R        : https://huggingface.co/sulaimank/wav2vec-xlsr-cv-grain-lg_both
  Wav2vec2-BERT: https://huggingface.co/sulaimank/w2v-bert-grain-lg_GRAIN
  Whisper      : https://huggingface.co/sulaimank/whisper-small-lug-grain

Age groups in metadata.csv: 18-29, 30-39, 40-49, 50-59, 60-69, 70-79
For Table 9, groups 50-59, 60-69, 70-79 are merged into >=50
as described in manuscript Section 5.1.

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
TARGET_SR  = 16000

MODEL_PATHS = {
    "xlsr"        : "xlsr/best_model",
    "wav2vec2bert": "wav2vec2bert/best_model",
    "whisper"     : "whisper/best_model",
}

AGE_REMAP = {
    "18-29": "18-29", "30-39": "30-39", "40-49": "40-49",
    "50-59": ">=50",  "60-69": ">=50",  "70-79": ">=50",
}
REGIONS  = ["Central", "Eastern", "Northern", "Western"]
GENDERS  = ["Female", "Male"]
AGE_EVAL = ["18-29", "30-39", "40-49", ">=50"]


# ── dataset builder ───────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, audio_dir: Path,
                  split: str = None) -> Dataset:
    if split:
        df = df[df["split"] == split].reset_index(drop=True)
    ds = Dataset.from_dict({
        "path"          : [str(audio_dir / r) for r in df["voice_clip"]],
        "sentence"      : df["sentence"].tolist(),
        "gender"        : df["gender"].tolist(),
        "age_group"     : [AGE_REMAP.get(a, a) for a in df["age_group"]],
        "Region"        : df["Region"].tolist(),
        "contributor_id": df["contributor_id"].tolist(),
    })
    return ds.cast_column("path", Audio(sampling_rate=TARGET_SR))


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe(ds: Dataset, model_path: str, batch_size: int = 16) -> list:
    """
    Run ASR inference using a HuggingFace pipeline.
    Works for XLS-R, Wav2vec2-BERT, and Whisper models.
    """
    from transformers import pipeline

    pipe   = pipeline("automatic-speech-recognition", model=model_path,
                      device=0 if torch.cuda.is_available() else -1,
                      chunk_length_s=30)
    arrays = [row["path"]["array"] for row in ds]
    preds  = []

    for i in range(0, len(arrays), batch_size):
        batch  = arrays[i: i + batch_size]
        result = pipe(batch, batch_size=batch_size)
        preds.extend([r["text"].lower().strip() for r in result])
        log.info("  %d / %d transcribed", min(i + batch_size, len(arrays)), len(arrays))

    return preds


def wer_pct(preds: list, refs: list) -> float:
    return round(wer_metric.compute(predictions=preds, references=refs) * 100, 2)


# ── table generators ──────────────────────────────────────────────────────────

def table6(test_ds: Dataset, model_dir: Path) -> pd.DataFrame:
    """Table 6 — WER for each fine-tuned model on the test set."""
    refs = [s.lower().strip() for s in test_ds["sentence"]]
    rows = []
    for key, rel in MODEL_PATHS.items():
        log.info("[Table 6] %s ...", key)
        preds = transcribe(test_ds, str(model_dir / rel))
        wer   = wer_pct(preds, refs)
        rows.append({"Fine-tuned ASR Model": key, "WER (%)": wer})
        log.info("  WER = %.2f%%", wer)
    return pd.DataFrame(rows)


def table7(our_test: Dataset, cv_ds: Dataset,
           model_dir: Path) -> pd.DataFrame:
    """Table 7 — Cross-dataset WER comparison."""
    best = str(model_dir / MODEL_PATHS["xlsr"])
    cv   = "facebook/wav2vec2-xls-r-300m"   # pre-existing baseline

    configs = [
        ("XLS-R wav2vec2-300M (fine-tuned)", best, our_test, "Our data",     "Our data"),
        ("XLS-R wav2vec2-300M (fine-tuned)", best, cv_ds,    "Our data",     "Common Voice"),
        ("XLS-R wav2vec2 (53 languages)",    cv,   our_test, "Common Voice", "Our data"),
        ("XLS-R wav2vec2 (53 languages)",    cv,   cv_ds,    "Common Voice", "Common Voice"),
    ]
    rows = []
    for name, path, ds, train_d, test_d in configs:
        log.info("[Table 7] %s | train=%s test=%s", name, train_d, test_d)
        preds = transcribe(ds, path)
        refs  = [s.lower().strip() for s in ds["sentence"]]
        rows.append({"Model": name, "Training Data": train_d,
                     "Test Data": test_d, "WER (%)": wer_pct(preds, refs)})
    return pd.DataFrame(rows)


def table8(our_test: Dataset, cv_ds: Dataset,
           model_path: str) -> pd.DataFrame:
    """Table 8 — WER by gender."""
    rows = []
    for ds, label in [(our_test, "Our data"), (cv_ds, "Common Voice")]:
        preds = transcribe(ds, model_path)
        refs  = [s.lower().strip() for s in ds["sentence"]]
        row   = {"Dataset": label}
        for g in GENDERS:
            idx  = [i for i, x in enumerate(ds["gender"]) if x == g]
            row[f"{g} WER (%)"] = wer_pct(
                [preds[i] for i in idx], [refs[i] for i in idx])
            log.info("[Table 8] %s | %s WER=%.2f%%  n=%d",
                     label, g, row[f"{g} WER (%)"], len(idx))
        rows.append(row)
    return pd.DataFrame(rows)


def table9(test_ds: Dataset, model_path: str) -> pd.DataFrame:
    """Table 9 — WER by gender and age group (50+ merged)."""
    preds   = transcribe(test_ds, model_path)
    refs    = [s.lower().strip() for s in test_ds["sentence"]]
    genders = test_ds["gender"]
    ages    = test_ds["age_group"]   # already remapped in build_dataset

    rows = []
    for g in GENDERS:
        for a in AGE_EVAL:
            idx = [i for i, (gi, ai) in enumerate(zip(genders, ages))
                   if gi == g and ai == a]
            if not idx:
                continue
            wer = wer_pct([preds[i] for i in idx], [refs[i] for i in idx])
            rows.append({"Gender": g, "Age Group": a, "WER (%)": wer})
            log.info("[Table 9] %s | %s  WER=%.2f%%  n=%d", g, a, wer, len(idx))
    return pd.DataFrame(rows)


def table10(test_ds: Dataset, model_path: str) -> pd.DataFrame:
    """Table 10 — WER by Ugandan regional accent."""
    preds   = transcribe(test_ds, model_path)
    refs    = [s.lower().strip() for s in test_ds["sentence"]]
    regions = test_ds["Region"]

    rows = []
    for r in REGIONS:
        idx = [i for i, x in enumerate(regions) if x == r]
        wer = wer_pct([preds[i] for i in idx], [refs[i] for i in idx])
        rows.append({"Regional Accent": f"{r} Uganda", "Average WER (%)": wer})
        log.info("[Table 10] %s  WER=%.2f%%  n=%d", r, wer, len(idx))
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
    parser.add_argument("--cv_audio_dir", default=None)
    parser.add_argument("--cv_metadata",  default=None)
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    model_dir  = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(args.metadata, encoding="utf-8")
    test_ds = build_dataset(df, audio_dir, split="test")
    log.info("Test set: %d clips", len(test_ds))

    cv_ds = None
    if args.cv_audio_dir and args.cv_metadata:
        cv_df = pd.read_csv(args.cv_metadata, encoding="utf-8")
        cv_ds = build_dataset(cv_df, Path(args.cv_audio_dir))
        log.info("Common Voice: %d clips", len(cv_ds))

    best_model = str(model_dir / MODEL_PATHS["xlsr"])

    t6 = table6(test_ds, model_dir)
    t6.to_csv(output_dir / "table6_model_wer.csv", index=False)
    print("\nTable 6:\n", t6.to_string(index=False))

    if cv_ds:
        t7 = table7(test_ds, cv_ds, model_dir)
        t7.to_csv(output_dir / "table7_cross_dataset_wer.csv", index=False)
        print("\nTable 7:\n", t7.to_string(index=False))

        t8 = table8(test_ds, cv_ds, best_model)
        t8.to_csv(output_dir / "table8_gender_wer.csv", index=False)
        print("\nTable 8:\n", t8.to_string(index=False))
    else:
        log.info("Skipping Tables 7 and 8 (no Common Voice data provided)")

    t9 = table9(test_ds, best_model)
    t9.to_csv(output_dir / "table9_age_gender_wer.csv", index=False)
    print("\nTable 9:\n", t9.to_string(index=False))

    t10 = table10(test_ds, best_model)
    t10.to_csv(output_dir / "table10_regional_wer.csv", index=False)
    print("\nTable 10:\n", t10.to_string(index=False))

    log.info("All results saved to %s", output_dir)
