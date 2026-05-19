# GRAIN Luganda Data

Preprocessing, training, and evaluation scripts for:

> **A Gender-Balanced and Region-Diverse Speech Corpus for Luganda**  
> Peter Nabende, Sulaiman Kagumire, Carol Kantono, Joyce Nakatumba-Nabende  
> *Data in Brief*, 2026

Dataset on Mendeley Data: **DOI [10.17632/gnsnrggr6m.2](https://data.mendeley.com/datasets/gnsnrggr6m/2)**

---

## Files

| Script | Manuscript section | What it does |
|---|---|---|
| `preprocess.py` | Section 4.4, 5.1 | Format conversion, VAD trimming, metadata alignment |
| `train.py` | Section 5.2 | Fine-tune XLS-R, Wav2vec2-BERT, Parakeet CTC |
| `evaluate.py` | Section 5.3, 5.4 | Generate Tables 6–10 (WER + bias assessment) |

---

## Requirements

```bash
pip install pandas librosa webrtcvad transformers datasets \
            evaluate jiwer torch torchaudio
```

`ffmpeg` must be on your PATH:

```bash
sudo apt install ffmpeg      # Ubuntu/Debian
brew install ffmpeg          # macOS
```

For Parakeet CTC only:

```bash
pip install nemo_toolkit['asr'] pytorch-lightning soundfile
```

---

## Dataset structure

Download from Mendeley Data. You will get:

```
audios.zip          — 21,858 WAV files (16 kHz mono 16-bit)
metadata.csv        — one row per recording
```

**metadata.csv columns:**

| Column | Type | Description |
|---|---|---|
| `sentence` | string (UTF-8) | Luganda transcription |
| `language` | string | Always "Luganda" |
| `contributor_id` | integer | Speaker ID (133 unique speakers) |
| `gender` | string | "Female" or "Male" |
| `age_group` | string | "18-29", "30-39", "40-49", "50-59", "60-69", "70-79" |
| `voice_clip` | string | WAV filename, e.g. `yogera_text_audio_20240425_113518.961214_2374.wav` |
| `duration` | float | Duration in **hours** (e.g. 0.001667 ≈ 6 s) |
| `Region` | string | "Central", "Eastern", "Northern", "Western" |

**Sample row:**

```
sentence       : Mu buganda abakazi batono abakama ente.
language       : Luganda
contributor_id : 193
gender         : Female
age_group      : 18-29
voice_clip     : yogera_text_audio_20240425_113518.961214_2374.wav
duration       : 0.000833
Region         : Central
```

---

## Step 1 — Preprocess

Converts all audio to 16 kHz mono 16-bit WAV, trims silence using
pywebrtcvad, validates duration (1–20 s), and aligns metadata.

```bash
python preprocess.py \
    --audio_dir   /path/to/extracted/audios \
    --metadata    /path/to/metadata.csv \
    --output_dir  /path/to/cleaned_audio \
    --output_meta /path/to/cleaned_audio/metadata.csv
```

Output:
- Cleaned WAV files (original filenames preserved)
- Updated `metadata.csv` with recomputed durations

---

## Step 2 — Train

Assigns train / validation / test splits (10,240 / 2,560 / 6,400)
and fine-tunes the selected model.

```bash
# Best-performing model only (recommended)
python train.py \
    --audio_dir  /path/to/cleaned_audio \
    --metadata   /path/to/cleaned_audio/metadata.csv \
    --output_dir /path/to/checkpoints \
    --model      xlsr

# All three models
python train.py ... --model all
```

**Model options:**

| `--model` | Model | HuggingFace / NeMo ID |
|---|---|---|
| `xlsr` | XLS-R wav2vec2-300M | `facebook/wav2vec2-xls-r-300m` |
| `wav2vec2bert` | Wav2vec2-BERT-2.0 | `facebook/w2v-bert-2.0` |
| `parakeet` | Parakeet-CTC-0.6B | `nvidia/parakeet-ctc-0.6b` |

**Training hyperparameters (manuscript Section 5.2):**

| Parameter | Value |
|---|---|
| GPU | NVIDIA A100 80 GB PCIe |
| Epochs | 100 |
| Learning rate | 0.0003 |
| Batch size | 32 |
| Random seed | 42 |

Output includes `metadata_with_splits.csv` needed by `evaluate.py`.

---

## Step 3 — Evaluate

Runs inference on the test split and saves all bias assessment tables.

```bash
python evaluate.py \
    --audio_dir    /path/to/cleaned_audio \
    --metadata     /path/to/checkpoints/metadata_with_splits.csv \
    --model_dir    /path/to/checkpoints \
    --output_dir   /path/to/results \
    --cv_audio_dir /path/to/common_voice_luganda \   # optional
    --cv_metadata  /path/to/cv_metadata.csv          # optional
```

**Output CSV files:**

| File | Manuscript table |
|---|---|
| `table6_model_wer.csv` | Table 6 — WER for three models |
| `table7_cross_dataset_wer.csv` | Table 7 — Cross-dataset WER |
| `table8_gender_wer.csv` | Table 8 — WER by gender |
| `table9_age_gender_wer.csv` | Table 9 — WER by age group and gender |
| `table10_regional_wer.csv` | Table 10 — WER by regional accent |

Tables 7 and 8 require Common Voice Luganda. Download from:
https://commonvoice.mozilla.org/en/datasets

---

## Notes on age groups

The metadata uses six age groups: 18-29, 30-39, 40-49, 50-59, 60-69, 70-79.
For evaluation (Table 9), the three oldest groups are merged into `>=50`
because their individual sample sizes are small (manuscript Section 5.1).
This merging is done automatically in `evaluate.py`.

---

## Citation

```bibtex
@article{nabende2026luganda,
  title   = {A Gender-Balanced and Region-Diverse Speech Corpus for {Luganda}},
  author  = {Nabende, Peter and Kagumire, Sulaiman and Kantono, Carol and
             Nakatumba-Nabende, Joyce},
  journal = {Data in Brief},
  year    = {2026},
  doi     = {10.17632/gnsnrggr6m.2}
}
```

---

## Licence

Scripts: MIT  
Dataset: CC BY 4.0
