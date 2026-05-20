#!/usr/bin/env python3
"""
Model 4: NLP Classification — Metadata-Conditioned LSTM
=========================================================
Single model that predicts effectiveness_3class from review text AND
urlDrugName + condition embeddings, so it learns drug/condition context
without needing separate fine-tuned models per combination.

Architecture:
    review text  → biLSTM  → hidden state ─┐
    urlDrugName  → Embedding ──────────────┤→ concat → Dropout → FC → prediction
    condition    → Embedding ──────────────┘

Phase 1: Reuse the pre-trained text checkpoint from train2.py if it exists,
         otherwise pre-train a plain LSTM on the full corpus and save it.
         Either way the text embedding + LSTM weights are transferred into the
         metadata model — only the drug/condition embeddings and the new FC
         layer train from scratch.
Phase 2: Train the full metadata-conditioned model on all 192k reviews.
 + attention over all LSTM timesteps (train_lstm0.py)
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (load_raw_data, clean_nlp_data, engineer_nlp_features,
                                          save_processed_data, load_processed_data)

PROCESSED_FILE   = "medication_feedback_processed.csv"
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
SAVED_MODEL_DIR  = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
PRETRAINED_CKPT  = SAVED_MODEL_DIR / "model_pretrained.pt"
PRETRAINED_VOCAB = SAVED_MODEL_DIR / "vocab_pretrained.joblib"
TARGET_COL       = "effectiveness_3class"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_REPO = "FrankAlonsoskyMolina/clearsight-analytics"
print(f"Using device: {DEVICE}")

def get_model_file(filename):
    """Return local path, downloading from HuggingFace if not found locally."""
    local_path = SAVED_MODEL_DIR / filename
    if not local_path.exists():
        print(f"{filename} not found locally — downloading from HuggingFace...")
        try:
            hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=str(SAVED_MODEL_DIR))
        except Exception as e:
            raise RuntimeError(
                f"Could not find '{filename}' locally or download it from "
                f"HuggingFace ({HF_REPO}).\nError: {e}"
            )
    return local_path


# =============================================================================
# Vocabulary
# =============================================================================

class Vocabulary:
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word2idx = {"<PAD>": 0, "<OOV>": 1}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common(self.max_words - 2):
            self.word2idx[word] = len(self.word2idx)
        return self

    def transform(self, texts, max_len=200):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(w, 1) for w in text.split()][:max_len]
            seq += [0] * (max_len - len(seq))
            sequences.append(seq)
        return np.array(sequences, dtype=np.int64)


# =============================================================================
# Dataset
# =============================================================================

class MetadataReviewDataset(Dataset):
    def __init__(self, X_text, X_drug, X_cond, y):
        self.text = torch.tensor(X_text, dtype=torch.long)
        self.drug = torch.tensor(X_drug, dtype=torch.long)
        self.cond = torch.tensor(X_cond, dtype=torch.long)
        self.y    = torch.tensor(y,      dtype=torch.long)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.drug[idx], self.cond[idx], self.y[idx]


# =============================================================================
# Plain LSTM — used only for Phase 1 pre-training (same as train2.py)
# =============================================================================

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=3, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


# =============================================================================
# Attention — learned weighted sum over LSTM timesteps
# =============================================================================

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):          # lstm_out: (batch, seq, hidden_dim)
        scores  = self.attn(lstm_out)     # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)
        return (lstm_out * weights).sum(dim=1)  # (batch, hidden_dim)


# =============================================================================
# Metadata-conditioned LSTM + Attention
# =============================================================================

class MetadataLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_drugs, num_conditions,
                 text_embed_dim=128, meta_embed_dim=32,
                 hidden_dim=128, num_layers=2, num_classes=3, dropout=0.4):
        super().__init__()
        self.text_embedding      = nn.Embedding(vocab_size,    text_embed_dim, padding_idx=0)
        self.drug_embedding      = nn.Embedding(num_drugs,     meta_embed_dim)
        self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
        self.lstm = nn.LSTM(
            text_embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 + meta_embed_dim * 2, num_classes)

    def forward(self, text, drug_idx, cond_idx):
        x = self.text_embedding(text)
        lstm_out, _ = self.lstm(x)              # (batch, seq, hidden*2)
        text_out = self.attention(lstm_out)     # (batch, hidden*2)
        drug_out = self.drug_embedding(drug_idx)
        cond_out = self.condition_embedding(cond_idx)
        combined = torch.cat([text_out, drug_out, cond_out], dim=1)
        return self.fc(self.dropout(combined))


# =============================================================================
# Data loading & preprocessing
# =============================================================================

def load_data() -> pd.DataFrame:
    if not (PROCESSED_DIR / PROCESSED_FILE).exists():
        print("Processed data not found — generating from raw...")
        df = load_raw_data("patient_medication_feedback.csv")
        df = clean_nlp_data(df)
        df = engineer_nlp_features(df)
        cols = ["condition", "urlDrugName", "effectiveness_3class",
                "benefitsReview", "review_text_clean",
                "review_word_count", "review_char_count"]
        save_processed_data(df[cols], PROCESSED_FILE)
    return load_processed_data(PROCESSED_FILE)


def preprocess_text(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def build_vocab(texts, max_words=10000):
    vocab = Vocabulary(max_words=max_words)
    vocab.fit(texts)
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, PRETRAINED_VOCAB)
    print(f"Vocabulary size: {len(vocab.word2idx)}")
    return vocab


def build_meta_encoders(df):
    """Fit LabelEncoders for urlDrugName and condition.

    'unknown' is always included as a class so unseen drugs/conditions at
    inference time can be safely mapped to it instead of raising an error.
    """
    drug_le = LabelEncoder()
    cond_le = LabelEncoder()
    drugs      = list(df["urlDrugName"].fillna("unknown").unique()) + ["unknown"]
    conditions = list(df["condition"].fillna("unknown").unique()) + ["unknown"]
    drug_le.fit(drugs)
    cond_le.fit(conditions)
    joblib.dump(drug_le, SAVED_MODEL_DIR / "drug_encoder.joblib")
    joblib.dump(cond_le, SAVED_MODEL_DIR / "condition_encoder.joblib")
    print(f"Unique drugs: {len(drug_le.classes_)}  |  Unique conditions: {len(cond_le.classes_)}")
    return drug_le, cond_le


def safe_encode(le: LabelEncoder, values) -> np.ndarray:
    """Encode values with a fallback to 'unknown' for anything not seen in training.

    Handles three cases automatically:
      - drug unknown, condition known  → drug maps to 'unknown' index
      - drug known,   condition unknown → condition maps to 'unknown' index
      - both unknown                   → both map to 'unknown' index
    In all cases the review text still contributes to the prediction.
    """
    known  = set(le.classes_)
    mapped = [v if v in known else "unknown" for v in values]
    return le.transform(mapped).astype(np.int64)


# =============================================================================
# Training helpers
# =============================================================================

def _run_epoch(model, loader, criterion, optimizer=None, scheduler=None, meta=True, desc=""):
    """One pass. Set meta=False for the plain LSTMClassifier (Phase 1)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        bar = tqdm(loader, desc=desc, leave=False)
        for batch in bar:
            if meta:
                text, drug, cond, y = [t.to(DEVICE) for t in batch]
                logits = model(text, drug, cond)
            else:
                text, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                logits = model(text)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def _fit(model, train_loader, val_loader, criterion, lr, epochs, patience, label,
         meta=True, use_scheduler=False):
    optimizer        = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss    = float("inf")
    patience_counter = 0
    best_weights     = None

    # OneCycleLR: linear warmup for 30% of steps, then cosine decay
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy="cos",
        )

    for epoch in range(epochs):
        t_loss = _run_epoch(model, train_loader, criterion, optimizer, scheduler=scheduler, meta=meta, desc=f"[{label}] Epoch {epoch+1}/{epochs} train")
        v_loss = _run_epoch(model, val_loader,   criterion, meta=meta, desc=f"[{label}] Epoch {epoch+1}/{epochs} val  ")
        current_lr = scheduler.get_last_lr()[0] if scheduler else lr
        print(f"  [{label}] Epoch {epoch+1}/{epochs} — train: {t_loss:.4f}  val: {v_loss:.4f}  lr: {current_lr:.2e}")

        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{label}] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    return model


# =============================================================================
# Phase 1 — plain LSTM pre-training (shared with train2.py)
# =============================================================================

class _TextOnlyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def pretrain_text_lstm(X_full, y_encoded, vocab_size):
    print("\n=== Phase 1: Pre-training text LSTM on full dataset ===")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    train_loader = DataLoader(_TextOnlyDataset(X_tr, y_tr),   batch_size=128, shuffle=True)
    val_loader   = DataLoader(_TextOnlyDataset(X_val, y_val), batch_size=128)

    model     = LSTMClassifier(vocab_size=vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    model = _fit(model, train_loader, val_loader, criterion,
                 lr=1e-3, epochs=10, patience=3, label="pretrain", meta=False)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PRETRAINED_CKPT)
    print(f"  Checkpoint saved to {PRETRAINED_CKPT}")
    return model


def transfer_text_weights(meta_model, ckpt_path):
    """Copy text embedding + LSTM weights from the plain LSTM checkpoint."""
    pretrained = torch.load(ckpt_path, map_location=DEVICE)
    own = meta_model.state_dict()
    mapping = {"embedding.weight": "text_embedding.weight"}
    transferred = 0
    for old_key, param in pretrained.items():
        new_key = mapping.get(old_key, old_key)   # remap embedding, keep lstm keys as-is
        if new_key in own and own[new_key].shape == param.shape:
            own[new_key].copy_(param)
            transferred += 1
    meta_model.load_state_dict(own)
    print(f"  Transferred {transferred} weight tensors from pre-trained checkpoint.")
    return meta_model


# =============================================================================
# Phase 2 — train metadata-conditioned model on full dataset
# =============================================================================

def train_meta_model(X_text, X_drug, X_cond, y_encoded,
                     vocab_size, num_drugs, num_conditions):
    print("\n=== Phase 2: Training metadata-conditioned LSTM on full dataset ===")
    (X_text_tr, X_text_val,
     X_drug_tr, X_drug_val,
     X_cond_tr, X_cond_val,
     y_tr, y_val) = train_test_split(
        X_text, X_drug, X_cond, y_encoded,
        test_size=0.1, random_state=42, stratify=y_encoded
    )

    train_loader = DataLoader(
        MetadataReviewDataset(X_text_tr, X_drug_tr, X_cond_tr, y_tr),
        batch_size=128, shuffle=True,
    )
    val_loader = DataLoader(
        MetadataReviewDataset(X_text_val, X_drug_val, X_cond_val, y_val),
        batch_size=128,
    )

    model = MetadataLSTMClassifier(
        vocab_size=vocab_size,
        num_drugs=num_drugs,
        num_conditions=num_conditions,
    ).to(DEVICE)

    if PRETRAINED_CKPT.exists():
        print("  Loading pre-trained text weights...")
        model = transfer_text_weights(model, PRETRAINED_CKPT)

    # Inverse-frequency class weights to address imbalance
    counts  = np.bincount(y_tr)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()   # normalise so loss scale stays stable
    print(f"  Class weights: {dict(zip(range(len(counts)), weights.cpu().numpy().round(4)))}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = _fit(model, train_loader, val_loader, criterion,
                 lr=1e-3, epochs=40, patience=4, label="meta-train", meta=True,
                 use_scheduler=True)

    print("Metadata-conditioned LSTM training complete.")
    return model


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_model(model, X_text, X_drug, X_cond, y_val, le, texts_val=None):
    model.eval()
    dataset = MetadataReviewDataset(
        X_text, X_drug, X_cond,
        np.zeros(len(X_text), dtype=np.int64),
    )
    loader = DataLoader(dataset, batch_size=256)

    all_preds = []
    with torch.no_grad():
        for text, drug, cond, _ in loader:
            logits = model(text.to(DEVICE), drug.to(DEVICE), cond.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    y_pred  = le.inverse_transform(all_preds)
    y_true  = np.array(y_val)
    classes = le.classes_

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix — Metadata-Conditioned LSTM")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    if texts_val is not None:
        print("\n--- Example Predictions ---")
        for text, actual, pred in zip(list(texts_val)[:5], list(y_true)[:5], list(y_pred)[:5]):
            print(f"  Text:      {text[:80]}...")
            print(f"  Actual:    {actual}")
            print(f"  Predicted: {pred}\n")


# =============================================================================
# Save
# =============================================================================

def save_artifacts(model, drug_le, cond_le, target_le):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVED_MODEL_DIR / "model_lstm0.pt")
    joblib.dump(drug_le,   SAVED_MODEL_DIR / "drug_encoder.joblib")
    joblib.dump(cond_le,   SAVED_MODEL_DIR / "condition_encoder.joblib")
    joblib.dump(target_le, SAVED_MODEL_DIR / "label_encoder_lstm0.joblib")
    print(f"\nSaved model and encoders to {SAVED_MODEL_DIR}")


# =============================================================================
# Main
# =============================================================================

def main():
    # 1. Load data
    df = load_data()
    y  = df[TARGET_COL]

    # 2. Vocab — use pre-cleaned text from processed CSV
    texts = df["review_text_clean"].fillna("").tolist()
    get_model_file("vocab_pretrained.joblib")
    if PRETRAINED_VOCAB.exists():
        print(f"\nLoading pre-built vocabulary from {PRETRAINED_VOCAB}")
        vocab = joblib.load(PRETRAINED_VOCAB)
    else:
        vocab = build_vocab(texts)

    X_text = vocab.transform(texts)

    # 3. Encode drug and condition
    drug_le, cond_le = build_meta_encoders(df)
    X_drug = safe_encode(drug_le, df["urlDrugName"].fillna("unknown"))
    X_cond = safe_encode(cond_le, df["condition"].fillna("unknown"))

    # 4. Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)

    # 5. Hold out 20% for final evaluation
    (X_text_train, X_text_test,
     X_drug_train, X_drug_test,
     X_cond_train, X_cond_test,
     y_train,      y_test,
     _,            texts_test) = train_test_split(
        X_text, X_drug, X_cond, y_encoded, texts,
        test_size=0.2, random_state=42, stratify=y_encoded,
    )

    vocab_size     = len(vocab.word2idx)
    num_drugs      = len(drug_le.classes_)
    num_conditions = len(cond_le.classes_)

    # 6. Phase 1 — pre-train plain LSTM (or reuse existing checkpoint)
    get_model_file("model_pretrained.pt")
    if not PRETRAINED_CKPT.exists():
        print("\nNo pre-trained checkpoint found — running Phase 1.")
        pretrain_text_lstm(X_text_train, y_train, vocab_size)
    else:
        print(f"\nPre-trained checkpoint found at {PRETRAINED_CKPT} — skipping Phase 1.")

    # 7. Phase 2 — train metadata-conditioned model
    model = train_meta_model(
        X_text_train, X_drug_train, X_cond_train, y_train,
        vocab_size, num_drugs, num_conditions,
    )

    # 8. Evaluate on held-out test set
    print("\n=== Evaluation on held-out test set ===")
    evaluate_model(
        model,
        X_text_test, X_drug_test, X_cond_test,
        target_le.inverse_transform(y_test),
        target_le,
        texts_val=texts_test,
    )

    # 9. Save
    save_artifacts(model, drug_le, cond_le, target_le)
    print("Training complete!")


if __name__ == "__main__":
    main()
