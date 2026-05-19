#!/usr/bin/env python3
"""
Model 4: NLP Classification — PyTorch LSTM
===========================================
LSTM text classifier using PyTorch on patient medication feedback.
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

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TARGET_COL = "effectiveness_3class"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# Vocabulary — builds word->index mapping from training texts
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
# Model
# =============================================================================

class TextCNNClassifier(nn.Module):
    """1D convolutions over embeddings with multiple kernel sizes, then max-pool."""
    def __init__(self, vocab_size, embed_dim=128, num_filters=128,
                 kernel_sizes=(2, 3, 4), num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)          # (batch, embed_dim, seq_len)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = self.dropout(torch.cat(x, dim=1))
        return self.fc(x)


# =============================================================================
# Pipeline functions
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load and clean the patient medication feedback dataset via the shared pipeline."""
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)
    return df


def preprocess_text(texts):
    """Lowercase, strip punctuation/digits, collapse whitespace.

    Returns a list of cleaned strings. Apply the SAME function at prediction time.
    """
    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def vectorize_text(texts, max_words=10000, max_len=200):
    """Build vocabulary and convert texts to padded integer sequences.

    Returns (X, vocab) — save vocab for prediction time.
    """
    vocab = Vocabulary(max_words=max_words)
    vocab.fit(texts)
    X = vocab.transform(texts, max_len=max_len)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, SAVED_MODEL_DIR / "vocab.joblib")
    print(f"Vocabulary size: {len(vocab.word2idx)} | Sequence matrix: {X.shape}")
    return X, vocab


def train_model(X_train, y_train, num_classes=3, vocab_size=10000,
                epochs=15, batch_size=64, patience=5):
    """Train an LSTM classifier using PyTorch.

    Returns (model, label_encoder).
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    # Split off a validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )

    train_loader = DataLoader(ReviewDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ReviewDataset(X_val, y_val), batch_size=batch_size)

    model = TextCNNClassifier(vocab_size=vocab_size, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    print("LSTM training complete.")
    return model, le


def evaluate_model(model, X_val, y_val, texts_val=None, le=None):
    """Evaluate the PyTorch LSTM model."""
    model.eval()
    dataset = ReviewDataset(X_val, np.zeros(len(X_val), dtype=np.int64))
    loader  = DataLoader(dataset, batch_size=256)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    y_pred = le.inverse_transform(all_preds)
    y_true = np.array(y_val)
    classes = le.classes_

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix — PyTorch LSTM")
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


def save_model(model, vocab):
    """Save the PyTorch model weights and vocabulary."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVED_MODEL_DIR / "model_pytorch.pt")
    joblib.dump(vocab, SAVED_MODEL_DIR / "vocab.joblib")
    print(f"Saved model and vocab to {SAVED_MODEL_DIR}")


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess text
    texts = preprocess_text(df["benefitsReview"])
    y = df[TARGET_COL]

    # 3. Vectorize
    X, vocab = vectorize_text(texts)

    # 4. Split
    X_train, X_val, y_train, y_val, _, texts_val = train_test_split(
        X, y, texts, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train
    model, le = train_model(X_train, y_train, vocab_size=len(vocab.word2idx))

    # 6. Evaluate
    evaluate_model(model, X_val, y_val, texts_val=texts_val, le=le)

    # 7. Save
    save_model(model, vocab)

    print("Training complete!")


if __name__ == "__main__":
    main()
