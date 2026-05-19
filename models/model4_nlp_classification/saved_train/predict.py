#!/usr/bin/env python3
"""
Model 4: NLP Classification — Prediction Script
=================================================
Loads your trained model and generates predictions on patient medication feedback.

Usage: python predict.py
Output: test_data/model4_results.csv
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

# =============================================================================
# ── Model Selection ───────────────────────────────────────────────────────────
# Change ACTIVE_MODEL to switch which trained model runs predictions.
#   "biobert_meta" → train4.py  (BioBERT + drug/condition metadata, PyTorch)
#   "meta_lstm"    → train3.py  (biLSTM + drug/condition metadata, PyTorch)
#   "plain_lstm"   → train2.py  (plain biLSTM, PyTorch)
#   "tfidf"        → train.py   (Logistic Regression + TF-IDF, sklearn)
ACTIVE_MODEL = "meta_lstm"

MODEL_CONFIGS = {
    "biobert_meta": {
        "model_file":       "model_biobert_meta.pt",
        "biobert_name":     "dmis-lab/biobert-base-cased-v1.2",
        "drug_enc_file":    "drug_encoder.joblib",
        "cond_enc_file":    "condition_encoder.joblib",
        "label_enc_file":   "label_encoder_biobert.joblib",
        "max_len":          256,
        "meta_embed_dim":   32,
        "hidden_dim":       256,
        "dropout":          0.3,
    },
    "meta_lstm": {
        "model_file":       "model_lstm0.pt",
        "vocab_file":       "vocab_pretrained.joblib",
        "drug_enc_file":    "drug_encoder.joblib",
        "cond_enc_file":    "condition_encoder.joblib",
        "label_enc_file":   "label_encoder_lstm0.joblib",
        "max_len":          200,
        "embed_dim":        128,
        "meta_embed_dim":   32,
        "hidden_dim":       128,
        "num_layers":       2,
        "dropout":          0.4,
    },
    "plain_lstm": {
        "model_file":       "model_pretrained.pt",
        "vocab_file":       "vocab_pretrained.joblib",
        "label_enc_file":   "label_encoder_meta.joblib",
        "max_len":          200,
        "embed_dim":        128,
        "hidden_dim":       128,
        "num_layers":       2,
        "dropout":          0.4,
    },
    "tfidf": {
        "model_file":       "model.joblib",
        "vectorizer_file":  "vectorizer.joblib",
    },
}
# =============================================================================

MODEL_DIR    = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE  = TEST_DATA_DIR / "model4_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model architecture classes (must match train2.py / train3.py / train4.py exactly)
# =============================================================================

class BioBERTMetadataClassifier(nn.Module):
    def __init__(self, biobert_name, num_drugs, num_conditions,
                 meta_embed_dim=32, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.bert                = AutoModel.from_pretrained(biobert_name)
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


class _BioBERTDataset(Dataset):
    """Lazy-tokenizing dataset — tokenizes one text at a time to avoid a large upfront memory spike."""
    def __init__(self, texts, X_drug, X_cond, tokenizer, max_len):
        self.texts     = texts
        self.drug      = torch.tensor(X_drug, dtype=torch.long)
        self.cond      = torch.tensor(X_cond, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            self.drug[idx],
            self.cond[idx],
        )

class Vocabulary:
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word2idx = {"<PAD>": 0, "<OOV>": 1}

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
        scores  = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
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
        lstm_out, _ = self.lstm(x)
        text_out = self.attention(lstm_out)
        drug_out = self.drug_embedding(drug_idx)
        cond_out = self.condition_embedding(cond_idx)
        combined = torch.cat([text_out, drug_out, cond_out], dim=1)
        return self.fc(self.dropout(combined))


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


class _TextDataset(Dataset):
    def __init__(self, X_text):
        self.text = torch.tensor(X_text, dtype=torch.long)
    def __len__(self): return len(self.text)
    def __getitem__(self, idx): return self.text[idx]


class _MetaDataset(Dataset):
    def __init__(self, X_text, X_drug, X_cond):
        self.text = torch.tensor(X_text, dtype=torch.long)
        self.drug = torch.tensor(X_drug, dtype=torch.long)
        self.cond = torch.tensor(X_cond, dtype=torch.long)
    def __len__(self): return len(self.text)
    def __getitem__(self, idx): return self.text[idx], self.drug[idx], self.cond[idx]


# =============================================================================
# Core functions
# =============================================================================

def load_model():
    """Load model artifacts for ACTIVE_MODEL.

    Returns a bundle dict with everything predict() needs.
    """
    cfg = MODEL_CONFIGS[ACTIVE_MODEL]

    if ACTIVE_MODEL == "biobert_meta":
        drug_le  = joblib.load(MODEL_DIR / cfg["drug_enc_file"])
        cond_le  = joblib.load(MODEL_DIR / cfg["cond_enc_file"])
        label_le = joblib.load(MODEL_DIR / cfg["label_enc_file"])

        tokenizer = AutoTokenizer.from_pretrained(cfg["biobert_name"])
        model = BioBERTMetadataClassifier(
            biobert_name   = cfg["biobert_name"],
            num_drugs      = len(drug_le.classes_),
            num_conditions = len(cond_le.classes_),
            meta_embed_dim = cfg["meta_embed_dim"],
            hidden_dim     = cfg["hidden_dim"],
            dropout        = cfg["dropout"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / cfg["model_file"], map_location=DEVICE))
        model.eval()
        print(f"Loaded biobert_meta from {cfg['model_file']}")
        return {"model": model, "tokenizer": tokenizer, "drug_le": drug_le,
                "cond_le": cond_le, "label_le": label_le,
                "max_len": cfg["max_len"], "approach": "biobert_meta"}

    elif ACTIVE_MODEL == "meta_lstm":
        vocab     = joblib.load(MODEL_DIR / cfg["vocab_file"])
        drug_le   = joblib.load(MODEL_DIR / cfg["drug_enc_file"])
        cond_le   = joblib.load(MODEL_DIR / cfg["cond_enc_file"])
        label_le  = joblib.load(MODEL_DIR / cfg["label_enc_file"])

        model = MetadataLSTMClassifier(
            vocab_size      = len(vocab.word2idx),
            num_drugs       = len(drug_le.classes_),
            num_conditions  = len(cond_le.classes_),
            text_embed_dim  = cfg["embed_dim"],
            meta_embed_dim  = cfg["meta_embed_dim"],
            hidden_dim      = cfg["hidden_dim"],
            num_layers      = cfg["num_layers"],
            dropout         = cfg["dropout"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / cfg["model_file"], map_location=DEVICE))
        model.eval()
        print(f"Loaded meta_lstm from {cfg['model_file']}")
        return {"model": model, "vocab": vocab, "drug_le": drug_le,
                "cond_le": cond_le, "label_le": label_le,
                "max_len": cfg["max_len"], "approach": "meta_lstm"}

    elif ACTIVE_MODEL == "plain_lstm":
        vocab    = joblib.load(MODEL_DIR / cfg["vocab_file"])
        label_le = joblib.load(MODEL_DIR / cfg["label_enc_file"])

        model = LSTMClassifier(
            vocab_size  = len(vocab.word2idx),
            embed_dim   = cfg["embed_dim"],
            hidden_dim  = cfg["hidden_dim"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / cfg["model_file"], map_location=DEVICE))
        model.eval()
        print(f"Loaded plain_lstm from {cfg['model_file']}")
        return {"model": model, "vocab": vocab, "label_le": label_le,
                "max_len": cfg["max_len"], "approach": "plain_lstm"}

    elif ACTIVE_MODEL == "tfidf":
        model      = joblib.load(MODEL_DIR / cfg["model_file"])
        vectorizer = joblib.load(MODEL_DIR / cfg["vectorizer_file"])
        print(f"Loaded tfidf from {cfg['model_file']}")
        return {"model": model, "vectorizer": vectorizer, "approach": "tfidf"}

    else:
        raise ValueError(f"Unknown ACTIVE_MODEL: '{ACTIVE_MODEL}'. "
                         f"Choose from: {list(MODEL_CONFIGS)}")


def preprocess_text(texts):
    """Clean text — must match the preprocessing used in training."""
    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def _safe_encode(le, values):
    """Encode with fallback to 'unknown' for unseen labels."""
    known  = set(le.classes_)
    mapped = [v if v in known else "unknown" for v in values]
    return le.transform(mapped).astype(np.int64)


def predict(bundle, df):
    """Run the loaded model on df and return a results DataFrame.

    Returns DataFrame with columns: id, predicted_class, confidence.
    """
    approach = bundle["approach"]
    model    = bundle["model"]

    if approach == "biobert_meta":
        # BioBERT is cased and handles its own tokenization — skip preprocess_text
        tokenizer = bundle["tokenizer"]
        drug_le   = bundle["drug_le"]
        cond_le   = bundle["cond_le"]
        label_le  = bundle["label_le"]
        max_len   = bundle["max_len"]

        X_drug  = _safe_encode(drug_le, df["urlDrugName"].fillna("unknown"))
        X_cond  = _safe_encode(cond_le, df["condition"].fillna("unknown"))
        texts   = df["benefitsReview"].astype(str).tolist()

        dataset = _BioBERTDataset(texts, X_drug, X_cond, tokenizer, max_len)
        loader  = DataLoader(dataset, batch_size=32, num_workers=0)

        all_probs = []
        with torch.no_grad():
            for input_ids, attn_mask, drug, cond in loader:
                logits = model(input_ids.to(DEVICE), attn_mask.to(DEVICE),
                               drug.to(DEVICE), cond.to(DEVICE))
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        all_probs         = np.vstack(all_probs)
        predicted_classes = label_le.inverse_transform(all_probs.argmax(axis=1))
        confidences       = all_probs.max(axis=1)

    elif approach == "meta_lstm":
        cleaned  = preprocess_text(df["benefitsReview"])
        vocab    = bundle["vocab"]
        drug_le  = bundle["drug_le"]
        cond_le  = bundle["cond_le"]
        label_le = bundle["label_le"]
        max_len  = bundle["max_len"]

        X_text = vocab.transform(cleaned, max_len=max_len)
        X_drug = _safe_encode(drug_le, df["urlDrugName"].fillna("unknown"))
        X_cond = _safe_encode(cond_le, df["condition"].fillna("unknown"))

        dataset = _MetaDataset(X_text, X_drug, X_cond)
        loader  = DataLoader(dataset, batch_size=256)

        all_probs = []
        with torch.no_grad():
            for text, drug, cond in loader:
                logits = model(text.to(DEVICE), drug.to(DEVICE), cond.to(DEVICE))
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        all_probs         = np.vstack(all_probs)
        predicted_classes = label_le.inverse_transform(all_probs.argmax(axis=1))
        confidences       = all_probs.max(axis=1)

    elif approach == "plain_lstm":
        cleaned  = preprocess_text(df["benefitsReview"])
        vocab    = bundle["vocab"]
        label_le = bundle["label_le"]
        max_len  = bundle["max_len"]

        X_text  = vocab.transform(cleaned, max_len=max_len)
        dataset = _TextDataset(X_text)
        loader  = DataLoader(dataset, batch_size=256)

        all_probs = []
        with torch.no_grad():
            for text in loader:
                logits = model(text.to(DEVICE))
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        all_probs         = np.vstack(all_probs)
        predicted_classes = label_le.inverse_transform(all_probs.argmax(axis=1))
        confidences       = all_probs.max(axis=1)

    elif approach == "tfidf":
        cleaned           = preprocess_text(df["benefitsReview"])
        vectorizer        = bundle["vectorizer"]
        X                 = vectorizer.transform(cleaned)
        predicted_classes = model.predict(X)
        confidences       = model.predict_proba(X).max(axis=1)

    return pd.DataFrame({
        "id":              df["Patient ID"].values,
        "predicted_class": predicted_classes,
        "confidence":      np.round(confidences, 4),
    })


def main():
    print(f"Active model: {ACTIVE_MODEL}")

    bundle = load_model()

    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)

    results = predict(bundle, df)

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nPredictions saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(results)}")
    print(results["predicted_class"].value_counts())


if __name__ == "__main__":
    main()
