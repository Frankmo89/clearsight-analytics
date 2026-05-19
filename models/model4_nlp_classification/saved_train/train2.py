#!/usr/bin/env python3
"""
Model 4: NLP Classification — LSTM Pre-train → Fine-tune
=========================================================
Phase 1: Pre-train a bidirectional LSTM on all 192k medication reviews so the
         model learns general review language and sentiment.
         Weights are saved to saved_model/model_pretrained.pt and reused on
         subsequent runs — pre-training only ever runs once.
Phase 2: Fine-tune the pre-trained model on a subset defined by FILTER_CONDITION
         and FILTER_DRUG.  Change those two constants and re-run to test a
         different subset without repeating Phase 1.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import load_raw_data, clean_nlp_data, engineer_nlp_features

SAVED_MODEL_DIR  = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
PRETRAINED_CKPT  = SAVED_MODEL_DIR / "model_pretrained.pt"
PRETRAINED_VOCAB = SAVED_MODEL_DIR / "vocab_pretrained.joblib"
TARGET_COL       = "effectiveness_3class"

# ── Change these two lines to fine-tune on a different subset ──────────────
FILTER_CONDITION = "birth control"
FILTER_DRUG      = "etonogestrel"
# ──────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


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

class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# Model — bidirectional LSTM
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
# Data loading
# =============================================================================

def load_all_data() -> pd.DataFrame:
    """Load the full 192k dataset for pre-training."""
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)
    return df


def load_filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return only Etonogestrel / Birth Control rows for fine-tuning."""
    mask = (
        df["condition"].str.lower().str.strip().eq(FILTER_CONDITION) &
        df["urlDrugName"].str.lower().str.strip().eq(FILTER_DRUG)
    )
    filtered = df[mask].reset_index(drop=True)
    print(f"\nFiltered to {FILTER_DRUG!r} / {FILTER_CONDITION!r}: {len(filtered)} rows")
    print(filtered[TARGET_COL].value_counts())
    return filtered


def preprocess_text(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def build_vocab(texts, max_words=10000, max_len=200):
    """Fit vocabulary on the full dataset and return (X, vocab).

    Saves to PRETRAINED_VOCAB so subsequent runs can reload it without
    re-fitting on the full corpus.
    """
    vocab = Vocabulary(max_words=max_words)
    vocab.fit(texts)
    X = vocab.transform(texts, max_len=max_len)
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, PRETRAINED_VOCAB)
    print(f"Vocabulary size: {len(vocab.word2idx)} | Matrix: {X.shape}")
    return X, vocab


# =============================================================================
# Training helpers
# =============================================================================

def _run_epoch(model, loader, criterion, optimizer=None):
    """One pass over a DataLoader. Pass optimizer=None for eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            if training:
                optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def _fit(model, X_train, y_train, criterion, lr, epochs, batch_size, patience, label):
    """Generic train loop with early stopping. Returns best model weights."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    train_loader = DataLoader(ReviewDataset(X_tr, y_tr),   batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ReviewDataset(X_val, y_val), batch_size=batch_size)

    optimizer        = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss    = float("inf")
    patience_counter = 0
    best_weights     = None

    for epoch in range(epochs):
        t_loss = _run_epoch(model, train_loader, criterion, optimizer)
        v_loss = _run_epoch(model, val_loader,   criterion)
        print(f"  [{label}] Epoch {epoch+1}/{epochs} — train: {t_loss:.4f}  val: {v_loss:.4f}")

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
# Phase 1 — Pre-train on full dataset
# =============================================================================

def pretrain(X_full, y_full, vocab_size, le):
    """Train on all 192k reviews and save weights to PRETRAINED_CKPT."""
    print("\n=== Phase 1: Pre-training on full dataset ===")
    y_encoded = le.fit_transform(y_full)
    model     = LSTMClassifier(vocab_size=vocab_size, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    model = _fit(
        model, X_full, y_encoded, criterion,
        lr=1e-3, epochs=10, batch_size=128, patience=3,
        label="pretrain",
    )

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PRETRAINED_CKPT)
    print(f"Pre-training complete. Checkpoint saved to {PRETRAINED_CKPT}")
    return model


def load_pretrained(vocab_size, le, y_full):
    """Load pre-trained weights from checkpoint (skips Phase 1)."""
    print(f"\n=== Phase 1: Loading pre-trained checkpoint from {PRETRAINED_CKPT} ===")
    le.fit(y_full)
    model = LSTMClassifier(vocab_size=vocab_size, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=DEVICE))
    print("Checkpoint loaded — skipping pre-training.")
    return model


# =============================================================================
# Phase 2 — Fine-tune on Etonogestrel / Birth Control subset
# =============================================================================

def finetune(model, X_sub, y_sub, le):
    """Fine-tune the pre-trained model on the filtered subset."""
    print("\n=== Phase 2: Fine-tuning on Etonogestrel / Birth Control ===")
    y_encoded = le.transform(y_sub)

    # Weighted loss to handle class imbalance in the small subset
    counts  = np.bincount(y_encoded)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = _fit(
        model, X_sub, y_encoded, criterion,
        lr=1e-4, epochs=20, batch_size=32, patience=5,
        label="finetune",
    )
    print("Fine-tuning complete.")
    return model


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_model(model, X_val, y_val, le, texts_val=None):
    model.eval()
    dataset = ReviewDataset(X_val, np.zeros(len(X_val), dtype=np.int64))
    loader  = DataLoader(dataset, batch_size=256)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(DEVICE))
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
    plt.title("Confusion Matrix — Fine-tuned LSTM (Etonogestrel / Birth Control)")
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


def save_model(model):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{FILTER_DRUG.replace(' ', '_')}_{FILTER_CONDITION.replace(' ', '_')}"
    out  = SAVED_MODEL_DIR / f"model_finetuned_{slug}.pt"
    torch.save(model.state_dict(), out)
    print(f"\nSaved fine-tuned model to {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    # 1. Load full dataset and filtered subset
    df_full = load_all_data()
    df_sub  = load_filtered_data(df_full)

    y_full = df_full[TARGET_COL]
    y_sub  = df_sub[TARGET_COL]
    le     = LabelEncoder()

    # 2. Vocab — reuse pretrained vocab if it exists, else build from full corpus
    if PRETRAINED_VOCAB.exists():
        print(f"\nLoading pre-built vocabulary from {PRETRAINED_VOCAB}")
        vocab = joblib.load(PRETRAINED_VOCAB)
    else:
        texts_full = preprocess_text(df_full["benefitsReview"])
        _, vocab   = build_vocab(texts_full)

    texts_sub = preprocess_text(df_sub["benefitsReview"])
    X_sub     = vocab.transform(texts_sub)

    # 3. Hold out 20% of the subset — fine-tune only sees the other 80%
    X_sub_train, X_sub_test, y_sub_train, y_sub_test, _, texts_test = train_test_split(
        X_sub, y_sub, texts_sub, test_size=0.2, random_state=42, stratify=y_sub
    )

    # 4. Phase 1 — pre-train or load checkpoint
    vocab_size = len(vocab.word2idx)
    if PRETRAINED_CKPT.exists():
        model = load_pretrained(vocab_size, le, y_full)
    else:
        texts_full = preprocess_text(df_full["benefitsReview"])
        X_full     = vocab.transform(texts_full)
        model      = pretrain(X_full, y_full, vocab_size=vocab_size, le=le)

    # 5. Phase 2 — fine-tune on filtered subset
    model = finetune(model, X_sub_train, y_sub_train, le)

    # 6. Evaluate on the held-out subset test split
    print(f"\n=== Evaluation on held-out {FILTER_DRUG!r} / {FILTER_CONDITION!r} reviews ===")
    evaluate_model(model, X_sub_test, y_sub_test, le, texts_val=texts_test)

    # 7. Save fine-tuned model (pre-trained checkpoint is already saved separately)
    save_model(model)
    print("Training complete!")


if __name__ == "__main__":
    main()
