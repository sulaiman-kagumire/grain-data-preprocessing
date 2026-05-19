"""
train.py
========
Fine-tuning script for the Yogera Luganda ASR bias experiments.
Trains three models as described in manuscript Section 5.2:

  Model 1 — XLS-R wav2vec2-300M   (HuggingFace)
  Model 2 — Wav2vec2-BERT-2.0     (HuggingFace)
  Model 3 — Parakeet-CTC-0.6B     (NVIDIA NeMo)

Training configuration (manuscript Section 5.2):
  GPU        : NVIDIA A100 80 GB PCIe
  Epochs     : 100
  LR         : 0.0003
  Batch size : 32

Data split (manuscript Section 5.1):
  Train      : 10,240 clips
  Validation :  2,560 clips
  Test       :  6,400 clips
  Method     : random sampling (seed fixed for reproducibility)

Metadata schema (actual columns from metadata.csv):
  sentence, language, contributor_id, gender, age_group,
  voice_clip, duration (hours), Region

Usage:
    python train.py \
        --audio_dir  /path/to/cleaned_audio \
        --metadata   /path/to/metadata.csv \
        --output_dir /path/to/checkpoints \
        --model      xlsr

    # or train all three:
    python train.py ... --model all

Requirements:
    pip install transformers datasets evaluate jiwer torch torchaudio
    For Parakeet: pip install nemo_toolkit['asr'] pytorch-lightning soundfile
"""

import os
import json
import logging
import argparse
import tempfile
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, Audio
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
MODEL_IDS = {
    "xlsr"        : "facebook/wav2vec2-xls-r-300m",
    "wav2vec2bert": "facebook/w2v-bert-2.0",
    "parakeet"    : "nvidia/parakeet-ctc-0.6b",
}

N_TRAIN  = 10240
N_VAL    = 2560
# remaining 6,400 are test (used by evaluate.py)
EPOCHS   = 100
LR       = 3e-4
BATCH    = 32
SEED     = 42


# ── split assignment ──────────────────────────────────────────────────────────

def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """
    Randomly assign train/validation/test splits.
    Sizes: 10,240 / 2,560 / 6,400 as stated in manuscript Section 5.1.
    Sampling is across the full dataset without stratification.
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["split"] = "test"
    df.loc[:N_TRAIN - 1,               "split"] = "train"
    df.loc[N_TRAIN: N_TRAIN + N_VAL - 1, "split"] = "validation"
    log.info("Split assigned: train=%d  val=%d  test=%d",
             (df["split"] == "train").sum(),
             (df["split"] == "validation").sum(),
             (df["split"] == "test").sum())
    return df


# ── dataset builder ───────────────────────────────────────────────────────────

def build_hf_dataset(df: pd.DataFrame, audio_dir: Path,
                     split: str) -> Dataset:
    """
    Build a HuggingFace Dataset from a metadata split.
    The voice_clip column contains filenames like:
        yogera_text_audio_20240425_113518.961214_2374.wav
    """
    subset = df[df["split"] == split].reset_index(drop=True)
    data = {
        "path"          : [str(audio_dir / r) for r in subset["voice_clip"]],
        "sentence"      : subset["sentence"].tolist(),
        "gender"        : subset["gender"].tolist(),
        "age_group"     : subset["age_group"].tolist(),
        "Region"        : subset["Region"].tolist(),
        "contributor_id": subset["contributor_id"].tolist(),
    }
    ds = Dataset.from_dict(data)
    ds = ds.cast_column("path", Audio(sampling_rate=TARGET_SR))
    return ds


TARGET_SR = 16000


# ── HuggingFace training ──────────────────────────────────────────────────────

def preprocess_batch(batch, processor):
    audio = batch["path"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=True,
    )
    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    batch["labels"]       = labels.input_ids[0]
    return batch


def train_hf(model_key: str, train_ds: Dataset, val_ds: Dataset,
             output_dir: Path):
    from transformers import (
        AutoProcessor,
        AutoModelForCTC,
        TrainingArguments,
        Trainer,
        DataCollatorCTCWithPadding,
    )

    model_id  = MODEL_IDS[model_key]
    processor = AutoProcessor.from_pretrained(model_id)

    log.info("Preprocessing %s dataset ...", model_key)
    train_ds = train_ds.map(
        lambda b: preprocess_batch(b, processor),
        remove_columns=train_ds.column_names,
        desc="Preprocessing train",
    )
    val_ds = val_ds.map(
        lambda b: preprocess_batch(b, processor),
        remove_columns=val_ds.column_names,
        desc="Preprocessing validation",
    )

    model = AutoModelForCTC.from_pretrained(
        model_id,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()

    wer_metric    = evaluate.load("wer")
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    def compute_metrics(pred):
        pred_ids  = pred.predictions.argmax(-1)
        pred_str  = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(
            predictions=pred_str, references=label_str)}

    ckpt_dir = output_dir / model_key
    args = TrainingArguments(
        output_dir                  = str(ckpt_dir),
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH,
        per_device_eval_batch_size  = BATCH,
        learning_rate               = LR,
        warmup_steps                = 500,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        logging_steps               = 100,
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 4,
        seed                        = SEED,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = processor.feature_extractor,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )

    log.info("Training %s for %d epochs ...", model_id, EPOCHS)
    trainer.train()

    best = ckpt_dir / "best_model"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    log.info("Best model saved to %s", best)


# ── Parakeet / NeMo training ──────────────────────────────────────────────────

def train_parakeet(train_ds: Dataset, val_ds: Dataset, output_dir: Path):
    try:
        import nemo.collections.asr as nemo_asr
        import pytorch_lightning as pl
        import soundfile as sf
    except ImportError:
        log.error("Install NeMo: pip install nemo_toolkit['asr'] "
                  "pytorch-lightning soundfile")
        return

    def ds_to_manifest(ds: Dataset, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for row in ds:
                rec = {
                    "audio_filepath": row["path"]["path"],
                    "text"          : row["sentence"],
                    "duration"      : len(row["path"]["array"]) / TARGET_SR,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_manifest = os.path.join(tmpdir, "train.jsonl")
        val_manifest   = os.path.join(tmpdir, "val.jsonl")
        ds_to_manifest(train_ds, train_manifest)
        ds_to_manifest(val_ds,   val_manifest)

        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            MODEL_IDS["parakeet"]
        )
        model.setup_training_data({"manifest_filepath": train_manifest,
                                   "batch_size": BATCH, "num_workers": 4})
        model.setup_validation_data({"manifest_filepath": val_manifest,
                                     "batch_size": BATCH, "num_workers": 4})

        trainer = pl.Trainer(
            max_epochs  = EPOCHS,
            accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        )
        trainer.fit(model)

        best = output_dir / "parakeet" / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        model.save_to(str(best / "parakeet_luganda.nemo"))
        log.info("Parakeet model saved to %s", best)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ASR models on Yogera Luganda corpus")
    parser.add_argument("--audio_dir",  required=True,
                        help="Directory of cleaned WAV files")
    parser.add_argument("--metadata",   required=True,
                        help="Path to metadata.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for model checkpoints")
    parser.add_argument("--model",      required=True,
                        choices=["xlsr", "wav2vec2bert", "parakeet", "all"],
                        help="Model to train")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata, encoding="utf-8")
    df = assign_splits(df)

    # save split-annotated metadata so evaluate.py can use the same split
    split_meta_path = output_dir / "metadata_with_splits.csv"
    df.to_csv(split_meta_path, index=False, encoding="utf-8")
    log.info("Split metadata saved to %s", split_meta_path)

    train_ds = build_hf_dataset(df, audio_dir, "train")
    val_ds   = build_hf_dataset(df, audio_dir, "validation")

    models_to_train = (["xlsr", "wav2vec2bert", "parakeet"]
                       if args.model == "all" else [args.model])

    for key in models_to_train:
        log.info("=" * 60)
        log.info("Training: %s", MODEL_IDS[key])
        log.info("=" * 60)
        if key == "parakeet":
            train_parakeet(train_ds, val_ds, output_dir)
        else:
            train_hf(key, train_ds, val_ds, output_dir)

    log.info("All training complete.")
