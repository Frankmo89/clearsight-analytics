# Model 4: NLP Drug Effectiveness Classification

This repository contains two NLP models that classify patient medication reviews into three effectiveness categories: **Highly Effective**, **Somewhat Effective**, and **Ineffective**. The production model is **BioBERT** — a 110-million parameter transformer pretrained on PubMed and PMC biomedical literature, then fine-tuned on 192,482 patient reviews using LoRA (Low-Rank Adaptation) so that only 0.87% of parameters train, preventing overfitting while achieving 90% accuracy. It is paired with drug and condition embeddings so the same review language is interpreted differently depending on which medication is being discussed. The second model is a **Bidirectional LSTM with Attention**, a lighter-weight alternative that also reaches 90% accuracy by learning which words in each review carry the most signal, combined with the same drug and condition embedding approach.

Because the trained model weights are too large for GitHub (up to 435 MB each), all `.pt` and `.joblib` files are hosted on HuggingFace at [FrankAlonsoskyMolina/clearsight-models](https://huggingface.co/FrankAlonsoskyMolina/clearsight-models). Both `predict.py` and the train scripts automatically check for each required file before running — if a file is missing from the `saved_model/` folder it is downloaded from HuggingFace on the fly with no manual steps required. This means the repository can be cloned and run immediately without downloading any model files in advance.

Trained on 192,482 patient reviews spanning 3,570 unique drugs and 906 conditions.

| Class | Count | % of Total |
|---|---|---|
| Highly Effective | 93,763 | 48.7% |
| Somewhat Effective | 64,544 | 33.5% |
| Ineffective | 34,175 | 17.8% |

---

## Switching Between Models

To switch between BioBERT and the LSTM, open `predict.py` and change the `ACTIVE_MODEL` variable near the top of the file:

```python
# "biobert" → BioBERT LoRA (best, F1 0.9018) — default
# "lstm"    → Bidirectional LSTM + Attention (F1 0.8972)
ACTIVE_MODEL = "biobert"
```

Set it to `"lstm"` to use the LSTM instead. The correct model weights and encoders will be loaded automatically — and downloaded from HuggingFace if not already present locally. No other changes are needed.

---

## Models

Two model families were developed. BioBERT is the production model.

| Model | File | Accuracy | Weighted F1 |
|---|---|---|---|
| **BioBERT LoRA** (production) | `train.py` | 90% | **0.9018** |
| LSTM + Attention | `train_lstm.py` | 90% | 0.8972 |

---

## Running Predictions

```bash
python models/model4_nlp_classification/predict.py
```

Output: `test_data/model4_results.csv`

### Switching Between Models

Open `predict.py` and change the `ACTIVE_MODEL` variable near the top of the file:

```python
# "biobert" → BioBERT LoRA (best, F1 0.9018) — default
# "lstm"    → Bidirectional LSTM + Attention (F1 0.8972)
ACTIVE_MODEL = "biobert"
```

### Output Format

The output CSV has three columns:

| Column | Description |
|---|---|
| `id` | Patient ID from the source dataset |
| `predicted_class` | One of: `Highly Effective`, `Somewhat Effective`, `Ineffective` |
| `confidence` | Softmax probability of the predicted class (0.0 – 1.0) |

---

## Automatic Model Downloads (HuggingFace)

Model weights and encoders are stored on HuggingFace at
[FrankAlonsoskyMolina/clearsight-models](https://huggingface.co/FrankAlonsoskyMolina/clearsight-models)
because the files are too large for GitHub (up to 435 MB each).

**You do not need to download anything manually.** Both `predict.py` and the
train scripts check for each required file before running. If a file is missing
from `saved_model/`, it is downloaded automatically:

```
model_lstm0.pt not found locally — downloading from HuggingFace...
```

If a file cannot be found locally or downloaded, a clear error is raised:

```
RuntimeError: Could not find 'model_lstm0.pt' locally or download it from
HuggingFace (FrankAlonsoskyMolina/clearsight-models).
```

Files stored on HuggingFace:

| File | Used By | Description |
|---|---|---|
| `model_biobert_lora_all_combos.pt` | BioBERT predict/train | Final BioBERT model weights |
| `model_biobert_phase2.pt` | BioBERT train | Phase 2 domain adaptation checkpoint |
| `model_lstm0.pt` | LSTM predict/train | Final LSTM model weights |
| `model_pretrained.pt` | LSTM train | Phase 1 pretrained LSTM checkpoint |
| `vocab_pretrained.joblib` | LSTM predict/train | 10k-word vocabulary |
| `drug_encoder.joblib` | Both | Label encoder for 3,570 drug names |
| `condition_encoder.joblib` | Both | Label encoder for 906 conditions |
| `label_encoder_biobert_lora_all_combos.joblib` | BioBERT predict | Output class encoder |
| `label_encoder_lstm0.joblib` | LSTM predict | Output class encoder |

---

## BioBERT Model

### Architecture

BioBERT (`dmis-lab/biobert-base-cased-v1.2`) is a 110-million parameter
transformer pretrained by the DMIS Lab on PubMed abstracts and PMC full-text
biomedical articles. Our model extends it with drug and condition embeddings:

```
review text  → BioBERT mean-pool (768-dim) ─┐
urlDrugName  → Embedding (32-dim) ───────────┤→ LayerNorm → Linear → GELU → Dropout → FC (3 classes)
condition    → Embedding (32-dim) ───────────┘
```

BioBERT receives the **raw, unprocessed review text** — it handles its own
tokenization and benefits from casing and punctuation.

### Three-Phase Training

**Phase 1 — BioBERT Biomedical Pretraining**
Inherited from HuggingFace. The base model arrives already pretrained on
PubMed and PMC. No code needed here.

**Phase 2 — Domain Adaptation** (`model_biobert_phase2.pt`)
All 110M BioBERT parameters are fine-tuned on the full 192,482 patient reviews.
This adapts the model from formal biomedical language (PubMed abstracts) to
informal patient review language. Takes approximately 6 hours. The checkpoint is
saved so this phase is skipped on subsequent runs.

**Phase 3 — LoRA Task Specialization**
LoRA (Low-Rank Adaptation) adapters are applied to the Q and V attention
projections of every transformer layer. Only ~0.87% of parameters train
(~948k out of 110M), preventing overfitting while learning the 3-class
classification task across all drug/condition combinations. After training,
adapters are merged back into the base weights so inference requires no
special PEFT library.

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `MAX_LEN` | 256 | Maximum token length per review |
| `LORA_R` | 16 | LoRA adapter rank |
| `LORA_ALPHA` | 32 | LoRA scaling factor |
| `LORA_LR` | 1e-4 | Learning rate for LoRA adapter parameters |
| `HEAD_LR` | 5e-4 | Learning rate for classification head |
| `PHASE3_EPOCHS` | 8 | Maximum Phase 3 training epochs |
| `PHASE3_PATIENCE` | 3 | Early stopping patience (epochs without improvement) |
| `meta_embed_dim` | 32 | Dimension of drug/condition embeddings |
| `hidden_dim` | 256 | Projection layer hidden size |
| `dropout` | 0.3 | Dropout rate |

### Train Script Outputs

Running `train.py` (or `train_biobert.py`) produces:

| File | Description |
|---|---|
| `model_biobert_phase2.pt` | Phase 2 checkpoint — reused to skip the 6-hour domain adaptation on future runs |
| `model_biobert_lora_all_combos.pt` | Final merged model weights — used by `predict.py` |
| `label_encoder_biobert_lora_all_combos.joblib` | Maps model output indices to class names |
| `drug_encoder.joblib` | Encodes drug names to integer indices |
| `condition_encoder.joblib` | Encodes condition names to integer indices |
| `confusion_matrix_biobert_lora_all_combos.png` | Per-class confusion matrix on the held-out test set |

---

## LSTM Model

### Architecture

A bidirectional LSTM with an attention mechanism and drug/condition metadata
embeddings:

```
review text  → Embedding (128-dim) → biLSTM → Attention → (256-dim) ─┐
urlDrugName  → Embedding (32-dim) ─────────────────────────────────────┤→ Dropout → FC (3 classes)
condition    → Embedding (32-dim) ─────────────────────────────────────┘
```

The LSTM receives **preprocessed text** — lowercased, HTML entities removed,
non-alphabetic characters stripped. Preprocessing is handled automatically
by the data pipeline before the model sees the text.

The attention mechanism learns which words in each review carry the most
signal for the effectiveness classification, rather than relying solely on
the final LSTM hidden state.

### Two-Phase Training

**Phase 1 — Plain LSTM Pretraining** (`model_pretrained.pt`)
A plain bidirectional LSTM (no metadata) is pretrained on all 192,482 reviews
to learn general medication language. The text embedding and LSTM weights are
saved as a checkpoint. On future runs this phase is skipped automatically.

**Phase 2 — Metadata-Conditioned Training**
The pretrained text weights are transferred into the full metadata model.
Only the drug/condition embedding layers and the new classification head
train from scratch. The full model is then trained on all reviews using
OneCycleLR scheduling — 30% linear warmup followed by cosine decay — and
inverse-frequency class weighting to prevent the majority class (Highly
Effective, 48.7%) from dominating.

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `MAX_WORDS` | 10,000 | Vocabulary size |
| `MAX_LEN` | 200 | Maximum token sequence length |
| `text_embed_dim` | 128 | Word embedding dimension |
| `meta_embed_dim` | 32 | Drug/condition embedding dimension |
| `hidden_dim` | 128 | LSTM hidden size (256 effective, bidirectional) |
| `num_layers` | 2 | Number of LSTM layers |
| `dropout` | 0.4 | Dropout rate |

### Train Script Outputs

Running `train_lstm.py` produces:

| File | Description |
|---|---|
| `model_pretrained.pt` | Phase 1 checkpoint — reused to skip pretraining on future runs |
| `vocab_pretrained.joblib` | 10k-word vocabulary built from the training corpus |
| `model_lstm0.pt` | Final model weights — used by `predict.py` |
| `label_encoder_lstm0.joblib` | Maps model output indices to class names |
| `drug_encoder.joblib` | Encodes drug names to integer indices |
| `condition_encoder.joblib` | Encodes condition names to integer indices |
| `confusion_matrix_lstm0.png` | Per-class confusion matrix on the held-out test set |

---

## Data Pipeline

Both models share the same preprocessing pipeline before training or
prediction:

1. **`load_raw_data()`** — loads `patient_medication_feedback.csv`
2. **`clean_data()`** — drops nulls, standardizes the 5-class effectiveness
   rating into 3 classes (`effectiveness_3class`)
3. **`engineer_features()`** — adds `review_text_clean`: lowercase,
   HTML entities removed, non-alphabetic characters stripped

BioBERT uses the raw `benefitsReview` column. The LSTM uses `review_text_clean`.
The processed dataset is cached to `data/processed/medication_feedback_processed.csv`
and reused on subsequent runs.

---

## Model Evolution Timeline

### LSTM Iterations

**train1.py — Basic PyTorch LSTM**
First attempt. Pure text classifier with no knowledge of which drug or condition
a review was about. Gave a workable baseline but left significant signal
on the table.

**train2.py — Pre-train + Fine-tune**
Introduced two-phase training: pre-train a plain LSTM on all reviews, then
transfer weights into the final model. Established the checkpoint reuse pattern
used by all later LSTM versions.

**train3.py — Metadata-Conditioned LSTM**
Added drug and condition embeddings alongside the text. The model now learns
that the same review language means different things depending on the drug being
discussed. This single change improved performance across every architecture tested.

**train8.py — Refined Metadata LSTM**
Further tuning of the metadata LSTM with improved regularization.

**train_lstm0.py / train_lstm.py — Final LSTM (F1 0.8972)**
Added an attention mechanism over all LSTM timesteps so the model learns which
words matter most, rather than relying on the final hidden state alone. Added
OneCycleLR scheduling (30% warmup + cosine decay) and inverse-frequency class
weighting. This is the LSTM version in production.

**train_lstm1.py — 30k Vocabulary (Abandoned)**
Increasing the vocabulary from 10,000 to 30,000 was expected to reduce
out-of-vocabulary errors on rare medical terms. In practice, accuracy dropped
from 90% to 79% (F1 0.7914) because rare words do not appear often enough in
the training data for the model to learn meaningful embeddings. The tighter
10k vocabulary forces the model to generalize from frequently seen words,
which is more effective.

---

### BioBERT Iterations

**train4.py — BioBERT + [CLS] Token**
First BioBERT attempt. Used the [CLS] token as the sentence representation.
The [CLS] approach was suboptimal — mean pooling across all tokens captures
richer context.

**train5.py — DistilBERT + Metadata**
Tested DistilBERT (40% smaller than BERT) for faster training. Performance
was lower than full BioBERT.

**train6.py — SentimentBERT + Metadata**
Tested a BERT model pretrained on Amazon and Yelp reviews, reasoning that
patient reviews resemble consumer reviews. The domain mismatch with medical
content meant BioBERT's biomedical pretraining was more valuable.

**train7.py — BioBERT with Differential Learning Rates**
Applied different learning rates to each transformer layer (lower rates for
earlier layers, higher for later layers), a standard technique for fine-tuning
transformers. Improved stability over uniform learning rates.

**train_biobert1.py — Full Fine-tune, Single Drug/Condition Combo (F1 0.8893)**
Fine-tuned all 110M BioBERT parameters on a single drug/condition combination
(etonogestrel / birth control, ~3,900 rows). Hit a ceiling quickly — 110M
parameters easily overfits on 3,900 examples.

**train_biobert2.py — LoRA, Single Combo (F1 0.8930)**
Introduced LoRA: froze all base weights, added rank-16 adapters to Q and V
attention projections. Only 0.87% of parameters train (~948k vs 110M).
Prevented overfitting and beat the full fine-tune on the same task.

**train_biobert3.py — Per-Combo LoRA (Abandoned)**
Attempted separate LoRA adapters per drug/condition combination (190 combos).
Produced 190 separate model files — operationally unmanageable and
inconsistent with the single-model approach the LSTM used.

**train_biobert4.py — Full Fine-tune, All Combos (Abandoned)**
Full fine-tuning on all 192k rows with no filtering. Approximately 2 hours
per epoch made iteration impractical.

**train_biobert5.py / train.py — LoRA, All Combos (F1 0.9018) — PRODUCTION**
Combined everything that worked: Phase 2 domain adaptation checkpoint reused,
LoRA on the full 192k dataset, single unified model. Drug and condition
embeddings differentiate all combinations while LoRA keeps training efficient
and prevents overfitting. This is the production model.

**train_biobert6.py — LoRA + OneCycleLR + Gradient Accumulation (F1 0.9010)**
Added OneCycleLR scheduling and gradient accumulation (effective batch size 128)
to the BioBERT5 setup. F1 of 0.9010 — essentially identical to BioBERT5
(0.9018). The bottleneck is the inherent ambiguity in patient-written reviews,
not the learning rate schedule.

---

### Ensemble Experiments

After both final models were trained, we tested combining their outputs:

**Soft Voting (55% BioBERT / 45% LSTM):** F1 0.9005 — marginally below
BioBERT alone.

**Logistic Regression Stacker:** F1 0.8637 — worse, caused by different
train/test splits between the two models (80/20 for LSTM, 90/10 for BioBERT),
which poisoned the stacker's training signal.

**Conclusion:** Ensembling did not improve on BioBERT5 alone. The complementary
strengths of the two models (LSTM stronger on Highly Effective, BioBERT
stronger on Ineffective) were not enough to overcome the split mismatch and
coordination overhead.