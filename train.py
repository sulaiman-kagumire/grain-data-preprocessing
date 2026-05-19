"""
train.py
========
Fine-tuning script for the Yogera Luganda ASR bias experiments.
Trains three models as described in manuscript Section 5.2:

  Model 1 — XLS-R wav2vec2-300M   (HuggingFace: facebook/wav2vec2-xls-r-300m)
  Model 2 — Wav2vec2-BERT-2.0     (HuggingFace: facebook/w2v-bert-2.0)
  Model 3 — Whisper-small         (HuggingFace: openai/whisper-small)

Fine-tuned models:
  XLS-R        : https://huggingface.co/sulaimank/wav2vec-xlsr-cv-grain-lg_both
  Wav2vec2-BERT: https://huggingface.co/sulaimank/w2v-bert-grain-lg_GRAIN
  Whisper      : https://huggingface.co/sulaimank/whisper-small-lug-grain

Training configuration (manuscript Section 5.2):
  GPU        : NVIDIA A100 80 GB PCIe
  Epochs     : 100
  LR         : 0.0003
  Batch size : 32

Data split (manuscript Section 5.1):
  Train      : 10,240 clips (~16.9 hours)
  Validation :  2,560 clips (~4.2 hours)
  Test       :  6,400 clips (~10.6 hours)
  Method     : random sampling (seed=42)

Usage:
    python train.py \
        --audio_dir  /path/to/cleaned_audio \
        --metadata   /path/to/metadata.csv \
        --output_dir /path/to/checkpoints \
        --model      xlsr

    # train all three models:
    python train.py ... --model all

Requirements:
    pip install transformers datasets evaluate jiwer torch torchaudio
"""

import logging
import argparse
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

MODEL_IDS = {
    "xlsr"        : "facebook/wav2vec2-xls-r-300m",
    "wav2vec2bert": "facebook/w2v-bert-2.0",
    "whisper"     : "openai/whisper-small",
}

N_TRAIN   = 10240
N_VAL     = 2560
EPOCHS    = 100
LR        = 3e-4
BATCH     = 32
SEED      = 42
TARGET_SR = 16000


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Random train/validation/test split: 10,240 / 2,560 / 6,400."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["split"] = "test"
    df.loc[:N_TRAIN - 1,                 "split"] = "train"
    df.loc[N_TRAIN: N_TRAIN + N_VAL - 1, "split"] = "validation"
    log.info("Split: train=%d  val=%d  test=%d",
             (df["split"] == "train").sum(),
             (df["split"] == "validation").sum(),
             (df["split"] == "test").sum())
    return df


def build_dataset(df: pd.DataFrame, audio_dir: Path, split: str) -> Dataset:
    """
    Build a HuggingFace Dataset for a given split.
    voice_clip filenames follow the Yogera convention:
        yogera_text_audio_20240425_113518.961214_2374.wav
    """
    sub = df[df["split"] == split].reset_index(drop=True)
    ds  = Dataset.from_dict({
        "path"          : [str(audio_dir / r) for r in sub["voice_clip"]],
        "sentence"      : sub["sentence"].tolist(),
        "gender"        : sub["gender"].tolist(),
        "age_group"     : sub["age_group"].tolist(),
        "Region"        : sub["Region"].tolist(),
        "contributor_id": sub["contributor_id"].tolist(),
    })
    return ds.cast_column("path", Audio(sampling_rate=TARGET_SR))


# ── CTC models: XLS-R and Wav2vec2-BERT ──────────────────────────────────────

def preprocess_ctc(batch, processor):
    audio  = batch["path"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"],
                       return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    batch["labels"]       = labels.input_ids[0]
    return batch


def train_ctc(model_key: str, train_ds: Dataset, val_ds: Dataset,
              output_dir: Path):
    from transformers import (
        AutoProcessor, AutoModelForCTC,
        TrainingArguments, Trainer, DataCollatorCTCWithPadding,
    )

    model_id  = MODEL_IDS[model_key]
    processor = AutoProcessor.from_pretrained(model_id)

    train_ds = train_ds.map(lambda b: preprocess_ctc(b, processor),
                            remove_columns=train_ds.column_names,
                            desc="Preprocessing train")
    val_ds   = val_ds.map(lambda b: preprocess_ctc(b, processor),
                          remove_columns=val_ds.column_names,
                          desc="Preprocessing val")

    model = AutoModelForCTC.from_pretrained(
        model_id, ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id)
    model.freeze_feature_encoder()

    wer_metric    = evaluate.load("wer")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        ids  = pred.predictions.argmax(-1)
        p    = processor.batch_decode(ids)
        lids = pred.label_ids
        lids[lids == -100] = processor.tokenizer.pad_token_id
        r    = processor.batch_decode(lids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=p, references=r)}

    ckpt = output_dir / model_key
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(ckpt), num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
            learning_rate=LR, warmup_steps=500,
            evaluation_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="wer",
            greater_is_better=False, logging_steps=100,
            fp16=torch.cuda.is_available(), dataloader_num_workers=4,
            seed=SEED, report_to="none"),
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )
    log.info("Training %s ...", model_id)
    trainer.train()
    best = ckpt / "best_model"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    log.info("Saved to %s", best)


# ── Whisper-small ─────────────────────────────────────────────────────────────

def preprocess_whisper(batch, processor):
    audio  = batch["path"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"],
                       return_tensors="pt")
    labels = processor.tokenizer(batch["sentence"], return_tensors="pt").input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"]         = labels[0]
    return batch


def train_whisper(train_ds: Dataset, val_ds: Dataset, output_dir: Path):
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
    )

    model_id  = MODEL_IDS["whisper"]
    processor = WhisperProcessor.from_pretrained(
        model_id, language="lg", task="transcribe")

    train_ds = train_ds.map(lambda b: preprocess_whisper(b, processor),
                            remove_columns=train_ds.column_names,
                            desc="Preprocessing train")
    val_ds   = val_ds.map(lambda b: preprocess_whisper(b, processor),
                          remove_columns=val_ds.column_names,
                          desc="Preprocessing val")

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="lg", task="transcribe")
    model.config.suppress_tokens = []

    wer_metric    = evaluate.load("wer")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, model=model, padding=True)

    def compute_metrics(pred):
        p_ids = pred.predictions
        l_ids = pred.label_ids
        l_ids[l_ids == -100] = processor.tokenizer.pad_token_id
        p = processor.tokenizer.batch_decode(p_ids, skip_special_tokens=True)
        r = processor.tokenizer.batch_decode(l_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=p, references=r)}

    ckpt = output_dir / "whisper"
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(ckpt), num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
            learning_rate=LR, warmup_steps=500,
            evaluation_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="wer",
            greater_is_better=False, logging_steps=100,
            fp16=torch.cuda.is_available(), predict_with_generate=True,
            generation_max_length=225, dataloader_num_workers=4,
            seed=SEED, report_to="none"),
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )
    log.info("Training Whisper-small ...")
    trainer.train()
    best = ckpt / "best_model"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    log.info("Saved to %s", best)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ASR models on Yogera Luganda corpus")
    parser.add_argument("--audio_dir",  required=True)
    parser.add_argument("--metadata",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model",      required=True,
                        choices=["xlsr", "wav2vec2bert", "whisper", "all"])
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata, encoding="utf-8")
    df = assign_splits(df)
    df.to_csv(output_dir / "metadata_with_splits.csv", index=False, encoding="utf-8")
    log.info("Split metadata saved.")

    train_ds = build_dataset(df, audio_dir, "train")
    val_ds   = build_dataset(df, audio_dir, "validation")

    for key in (["xlsr", "wav2vec2bert", "whisper"] if args.model == "all"
                else [args.model]):
        log.info("=" * 60)
        log.info("Training: %s", MODEL_IDS[key])
        log.info("=" * 60)
        if key == "whisper":
            train_whisper(train_ds, val_ds, output_dir)
        else:
            train_ctc(key, train_ds, val_ds, output_dir)

    log.info("All training complete.")
