#!/usr/bin/env python3
"""
Model 4: NLP Classification — Ensemble Prediction (BioBERT5 + LSTM)
====================================================================
Combines softmax probabilities from BioBERT5 and LSTM via weighted averaging.

    BioBERT5 weight: 0.55  (val weighted-F1 = 0.9018)
    LSTM weight:     0.45  (val weighted-F1 = 0.8972)

The two models have complementary strengths:
    Highly Effective  → LSTM stronger (0.93 vs 0.92)
    Ineffective       → BioBERT stronger (0.94 vs 0.90)
    Somewhat Effective → BioBERT stronger (0.86 vs 0.85)

Usage:
    python models/model4_nlp_classification/predict_ensemble.py

Output:
    test_data/model4_ensemble_results.csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import load_raw_data, clean_nlp_data, engineer_nlp_features

MODEL_DIR     = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE   = TEST_DATA_DIR / "model4_ensemble_results.csv"
BIOBERT_NAME  = "dmis-lab/biobert-base-cased-v1.2"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIOBERT_WEIGHT = 0.55
LSTM_WEIGHT    = 0.45


# =============================================================================
# Model architectures (must match training files exactly)
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
        combined    = torch.cat([text_out, drug_out, cond_out], dim=1)
        return self.fc(self.dropout(combined))


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
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def _safe_encode(le, values):
    known  = set(le.classes_)
    mapped = [v if v in known else "unknown" for v in values]
    return le.transform(mapped).astype(np.int64)


# =============================================================================
# Model loading
# =============================================================================

def load_biobert5():
    drug_le  = joblib.load(MODEL_DIR / "drug_encoder.joblib")
    cond_le  = joblib.load(MODEL_DIR / "condition_encoder.joblib")
    label_le = joblib.load(MODEL_DIR / "label_encoder_biobert_lora_all_combos.joblib")
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_NAME)

    model = BioBERTMetadataClassifier(
        num_drugs=len(drug_le.classes_), num_conditions=len(cond_le.classes_)
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "model_biobert_lora_all_combos.pt", map_location=DEVICE)
    )
    model.eval()
    print(f"Loaded BioBERT5: {len(drug_le.classes_)} drugs, {len(cond_le.classes_)} conditions")
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
    print(f"Loaded LSTM: vocab={len(vocab.word2idx)}, {len(drug_le.classes_)} drugs")
    return {"model": model, "vocab": vocab,
            "drug_le": drug_le, "cond_le": cond_le, "label_le": label_le}


# =============================================================================
# Inference
# =============================================================================

def _get_biobert_probs(bundle, df):
    tokenizer = bundle["tokenizer"]
    drug_le   = bundle["drug_le"]
    cond_le   = bundle["cond_le"]

    texts  = df["benefitsReview"].astype(str).tolist()
    X_drug = _safe_encode(drug_le, df["urlDrugName"].fillna("unknown"))
    X_cond = _safe_encode(cond_le, df["condition"].fillna("unknown"))

    dataset = _BioBERTDataset(texts, X_drug, X_cond, tokenizer)
    loader  = DataLoader(dataset, batch_size=32, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for ids, mask, drug, cond in loader:
            logits = bundle["model"](ids.to(DEVICE), mask.to(DEVICE),
                                     drug.to(DEVICE), cond.to(DEVICE))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


def _get_lstm_probs(bundle, df):
    vocab   = bundle["vocab"]
    drug_le = bundle["drug_le"]
    cond_le = bundle["cond_le"]

    cleaned = _preprocess_text(df["benefitsReview"])
    X_text  = vocab.transform(cleaned, max_len=200)
    X_drug  = _safe_encode(drug_le, df["urlDrugName"].fillna("unknown"))
    X_cond  = _safe_encode(cond_le, df["condition"].fillna("unknown"))

    dataset = _MetaDataset(X_text, X_drug, X_cond)
    loader  = DataLoader(dataset, batch_size=256)

    all_probs = []
    with torch.no_grad():
        for text, drug, cond in loader:
            logits = bundle["model"](text.to(DEVICE), drug.to(DEVICE), cond.to(DEVICE))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(all_probs)


def predict_ensemble(df, biobert_bundle, lstm_bundle):
    print("\nRunning BioBERT5 inference...")
    biobert_probs = _get_biobert_probs(biobert_bundle, df)

    print("Running LSTM inference...")
    lstm_probs = _get_lstm_probs(lstm_bundle, df)

    ensemble_probs    = BIOBERT_WEIGHT * biobert_probs + LSTM_WEIGHT * lstm_probs
    label_le          = biobert_bundle["label_le"]
    predicted_classes = label_le.inverse_transform(ensemble_probs.argmax(axis=1))
    confidences       = ensemble_probs.max(axis=1)

    return pd.DataFrame({
        "id":              df["Patient ID"].values,
        "predicted_class": predicted_classes,
        "confidence":      np.round(confidences, 4),
    })


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Ensemble weights — BioBERT5: {BIOBERT_WEIGHT}, LSTM: {LSTM_WEIGHT}\n")

    biobert_bundle = load_biobert5()
    lstm_bundle    = load_lstm()

    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)

    results = predict_ensemble(df, biobert_bundle, lstm_bundle)

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nEnsemble predictions saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(results)}")
    print(results["predicted_class"].value_counts())


if __name__ == "__main__":
    main()