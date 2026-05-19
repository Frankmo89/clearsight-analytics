#!/usr/bin/env python3
"""
Ensemble Stacker: BioBERT5 + LSTM → Logistic Regression meta-learner
=====================================================================
Trains a logistic regression on top of both models' softmax outputs.

    Input features (6 total):
        BioBERT5 softmax [p_highly_effective, p_ineffective, p_somewhat_effective]
        LSTM     softmax [p_highly_effective, p_ineffective, p_somewhat_effective]

    Meta-learner learns which model to trust per class, outperforming
    fixed-weight averaging.

Data split (mirrors train_biobert5.py):
    Full 192k → 10% test hold-out (random_state=42)
               → 10% of remainder = val (random_state=42)
    Stacker trained on val (17,324 rows), evaluated on test (19,249 rows).

    Note: The LSTM used a 20/80 split, so it may have seen some val/test
    rows during training. BioBERT5 strictly never saw them.

Usage:
    python models/model4_nlp_classification/train_ensemble_stacker.py

Outputs:
    saved_model/stacker_lr.joblib        ← meta-learner
    saved_model/stacker_label_le.joblib  ← label encoder
"""

import re
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import load_raw_data, clean_nlp_data, engineer_nlp_features

MODEL_DIR    = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
BIOBERT_NAME = "dmis-lab/biobert-base-cased-v1.2"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model architectures (must match training files)
# =============================================================================

class BioBERTMetadataClassifier(nn.Module):
    def __init__(self, num_drugs, num_conditions,
                 meta_embed_dim=32, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.bert                = AutoModel.from_pretrained(BIOBERT_NAME)
        self.drug_embedding      = nn.Embedding(num_drugs,      meta_embed_dim)
        self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
        combined_dim = 768 + meta_embed_dim * 2
        self.norm    = nn.LayerNorm(combined_dim)
        self.proj    = nn.Linear(combined_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1)

    def forward(self, input_ids, attention_mask, drug_idx, cond_idx):
        outputs  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_out = self._mean_pool(outputs.last_hidden_state, attention_mask)
        drug_out = self.drug_embedding(drug_idx)
        cond_out = self.condition_embedding(cond_idx)
        combined = torch.cat([text_out, drug_out, cond_out], dim=1)
        combined = self.norm(combined)
        combined = torch.nn.functional.gelu(self.proj(combined))
        return self.fc(self.dropout(combined))


class Vocabulary:
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word2idx  = {"<PAD>": 0, "<OOV>": 1}

    def transform(self, texts, max_len=200):
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(w, 1) for w in text.split()][:max_len]
            seq += [0] * (max_len - len(seq))
            sequences.append(seq)
        return np.array(sequences, dtype=np.int64)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return (lstm_out * weights).sum(dim=1)


class MetadataLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_drugs, num_conditions,
                 text_embed_dim=128, meta_embed_dim=32,
                 hidden_dim=128, num_layers=2, num_classes=3, dropout=0.4):
        super().__init__()
        self.text_embedding      = nn.Embedding(vocab_size,     text_embed_dim, padding_idx=0)
        self.drug_embedding      = nn.Embedding(num_drugs,      meta_embed_dim)
        self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
        self.lstm = nn.LSTM(
            text_embed_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 + meta_embed_dim * 2, num_classes)

    def forward(self, text, drug_idx, cond_idx):
        x = self.text_embedding(text)
        lstm_out, _ = self.lstm(x)
        text_out    = self.attention(lstm_out)
        drug_out    = self.drug_embedding(drug_idx)
        cond_out    = self.condition_embedding(cond_idx)
        return self.fc(self.dropout(torch.cat([text_out, drug_out, cond_out], dim=1)))


# =============================================================================
# Datasets
# =============================================================================

class _BioBERTDataset(Dataset):
    def __init__(self, texts, X_drug, X_cond, tokenizer, max_len=256):
        self.texts     = texts
        self.drug      = torch.tensor(X_drug, dtype=torch.long)
        self.cond      = torch.tensor(X_cond, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), \
               self.drug[idx], self.cond[idx]


class _MetaDataset(Dataset):
    def __init__(self, X_text, X_drug, X_cond):
        self.text = torch.tensor(X_text, dtype=torch.long)
        self.drug = torch.tensor(X_drug, dtype=torch.long)
        self.cond = torch.tensor(X_cond, dtype=torch.long)

    def __len__(self): return len(self.text)
    def __getitem__(self, idx): return self.text[idx], self.drug[idx], self.cond[idx]


# =============================================================================
# Helpers
# =============================================================================

def _preprocess_text(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower().replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        cleaned.append(re.sub(r"\s+", " ", text).strip())
    return cleaned


def _safe_encode(le, values):
    known  = set(le.classes_)
    mapped = [v if v in known else "unknown" for v in values]
    return le.transform(mapped).astype(np.int64)


# =============================================================================
# Model loading
# =============================================================================

def load_biobert5():
    drug_le   = joblib.load(MODEL_DIR / "drug_encoder.joblib")
    cond_le   = joblib.load(MODEL_DIR / "condition_encoder.joblib")
    label_le  = joblib.load(MODEL_DIR / "label_encoder_biobert_lora_all_combos.joblib")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_NAME)

    model = BioBERTMetadataClassifier(
        num_drugs=len(drug_le.classes_), num_conditions=len(cond_le.classes_)
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "model_biobert_lora_all_combos.pt", map_location=DEVICE)
    )
    model.eval()
    print(f"Loaded BioBERT5")
    return {"model": model, "tokenizer": tokenizer,
            "drug_le": drug_le, "cond_le": cond_le, "label_le": label_le}


def load_lstm():
    vocab    = joblib.load(MODEL_DIR / "vocab_pretrained.joblib")
    drug_le  = joblib.load(MODEL_DIR / "drug_encoder.joblib")
    cond_le  = joblib.load(MODEL_DIR / "condition_encoder.joblib")
    label_le = joblib.load(MODEL_DIR / "label_encoder_lstm0.joblib")

    model = MetadataLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        num_drugs=len(drug_le.classes_),
        num_conditions=len(cond_le.classes_),
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "model_lstm0.pt", map_location=DEVICE)
    )
    model.eval()
    print(f"Loaded LSTM")
    return {"model": model, "vocab": vocab,
            "drug_le": drug_le, "cond_le": cond_le, "label_le": label_le}


# =============================================================================
# Inference — returns raw softmax probabilities
# =============================================================================

def get_biobert_probs(bundle, df_subset):
    tokenizer = bundle["tokenizer"]
    drug_le   = bundle["drug_le"]
    cond_le   = bundle["cond_le"]

    texts  = df_subset["benefitsReview"].astype(str).tolist()
    X_drug = _safe_encode(drug_le, df_subset["urlDrugName"].fillna("unknown"))
    X_cond = _safe_encode(cond_le, df_subset["condition"].fillna("unknown"))

    loader = DataLoader(
        _BioBERTDataset(texts, X_drug, X_cond, tokenizer),
        batch_size=32, num_workers=0,
    )
    all_probs = []
    with torch.no_grad():
        for ids, mask, drug, cond in loader:
            logits = bundle["model"](ids.to(DEVICE), mask.to(DEVICE),
                                     drug.to(DEVICE), cond.to(DEVICE))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


def get_lstm_probs(bundle, df_subset):
    vocab   = bundle["vocab"]
    drug_le = bundle["drug_le"]
    cond_le = bundle["cond_le"]

    cleaned = _preprocess_text(df_subset["benefitsReview"])
    X_text  = vocab.transform(cleaned, max_len=200)
    X_drug  = _safe_encode(drug_le, df_subset["urlDrugName"].fillna("unknown"))
    X_cond  = _safe_encode(cond_le, df_subset["condition"].fillna("unknown"))

    loader = DataLoader(_MetaDataset(X_text, X_drug, X_cond), batch_size=256)
    all_probs = []
    with torch.no_grad():
        for text, drug, cond in loader:
            logits = bundle["model"](text.to(DEVICE), drug.to(DEVICE), cond.to(DEVICE))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}\n")

    # ── Load data (same pipeline as train_biobert5.py) ────────────────────────
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)

    target_le = LabelEncoder()
    y = target_le.fit_transform(df["effectiveness_3class"])

    # Exact same splits as train_biobert5.py
    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.1, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_train, test_size=0.1, random_state=42, stratify=y[idx_train])

    print(f"Split — train: {len(idx_train):,}  val: {len(idx_val):,}  test: {len(idx_test):,}")
    print(f"Stacker trains on val, evaluates on test.\n")

    df_val  = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    # ── Load base models ──────────────────────────────────────────────────────
    biobert_bundle = load_biobert5()
    lstm_bundle    = load_lstm()

    # ── Get softmax probs on val set (stacker training data) ─────────────────
    print("\nGetting BioBERT5 probs on val set...")
    bb_val  = get_biobert_probs(biobert_bundle, df_val)
    print("Getting LSTM probs on val set...")
    lstm_val = get_lstm_probs(lstm_bundle, df_val)
    X_val_stack = np.hstack([bb_val, lstm_val])   # (17324, 6)

    # ── Get softmax probs on test set (stacker evaluation) ───────────────────
    print("\nGetting BioBERT5 probs on test set...")
    bb_test   = get_biobert_probs(biobert_bundle, df_test)
    print("Getting LSTM probs on test set...")
    lstm_test = get_lstm_probs(lstm_bundle, df_test)
    X_test_stack = np.hstack([bb_test, lstm_test])  # (19249, 6)

    # ── Train logistic regression stacker ────────────────────────────────────
    print("\nTraining logistic regression stacker...")
    stacker = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
    )
    stacker.fit(X_val_stack, y_val)
    print("Stacker trained.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred_test = stacker.predict(X_test_stack)
    test_f1     = f1_score(y_test, y_pred_test, average="weighted")

    print(f"\n{'='*55}")
    print(f"  Stacker test weighted F1: {test_f1:.4f}")
    print(f"{'='*55}")
    print(classification_report(
        y_test, y_pred_test,
        target_names=target_le.classes_,
    ))

    # Compare baselines on same test split
    bb_pred   = bb_test.argmax(axis=1)
    lstm_pred = lstm_test.argmax(axis=1)
    print(f"BioBERT5 alone on this test split:  {f1_score(y_test, bb_pred,   average='weighted'):.4f}")
    print(f"LSTM alone on this test split:       {f1_score(y_test, lstm_pred, average='weighted'):.4f}")
    ensemble_probs = 0.55 * bb_test + 0.45 * lstm_test
    print(f"Soft-vote ensemble (55/45):          {f1_score(y_test, ensemble_probs.argmax(1), average='weighted'):.4f}")
    print(f"Stacker (LR meta-learner):           {test_f1:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(stacker,   MODEL_DIR / "stacker_lr.joblib")
    joblib.dump(target_le, MODEL_DIR / "stacker_label_le.joblib")
    print(f"\nSaved stacker to {MODEL_DIR / 'stacker_lr.joblib'}")


if __name__ == "__main__":
    main()