#!/usr/bin/env python3
"""
Model 4: NLP Classification — DistilBERT + Metadata (drug & condition embeddings)
==================================================================================
Same architecture as train4.py but uses DistilBERT (distilbert-base-uncased)
instead of BioBERT. DistilBERT is 40% smaller and 60% faster than BERT while
retaining ~97% of its performance.

Architecture:
    review text  → DistilBERT → [CLS] token (768-dim) ─┐
    urlDrugName  → Embedding (32-dim) ──────────────────┤→ concat → Dropout → FC → prediction
    condition    → Embedding (32-dim) ──────────────────┘

Improvements over base approach:
    - Inverse-frequency class weights on CrossEntropyLoss
    - OneCycleLR scheduler: 30% linear warmup then cosine decay
    - Differential learning rates: 2e-5 for DistilBERT, 1e-3 for new layers

Reuses drug_encoder.joblib and condition_encoder.joblib from train3.py if they exist.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import load_raw_data, clean_nlp_data, engineer_nlp_features
from models.model4_nlp_classification.train3 import safe_encode

SAVED_MODEL_DIR  = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TARGET_COL       = "effectiveness_3class"
DISTILBERT_MODEL = "distilbert-base-uncased"
MAX_LEN          = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# Dataset
# =============================================================================

class DistilBERTDataset(Dataset):
    def __init__(self, input_ids, attention_masks, X_drug, X_cond, y):
        self.input_ids       = input_ids
        self.attention_masks = attention_masks
        self.drug = torch.tensor(X_drug, dtype=torch.long)
        self.cond = torch.tensor(X_cond, dtype=torch.long)
        self.y    = torch.tensor(y,      dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.drug[idx],
            self.cond[idx],
            self.y[idx],
        )


# =============================================================================
# Model
# =============================================================================

class DistilBERTMetadataClassifier(nn.Module):
    def __init__(self, num_drugs, num_conditions,
                 meta_embed_dim=32, num_classes=3, dropout=0.3):
        super().__init__()
        self.distilbert          = AutoModel.from_pretrained(DISTILBERT_MODEL)
        self.drug_embedding      = nn.Embedding(num_drugs,      meta_embed_dim)
        self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 768 (DistilBERT hidden size) + 32 + 32
        self.fc = nn.Linear(768 + meta_embed_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask, drug_idx, cond_idx):
        outputs  = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out  = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        drug_out = self.drug_embedding(drug_idx)
        cond_out = self.condition_embedding(cond_idx)
        combined = torch.cat([cls_out, drug_out, cond_out], dim=1)
        return self.fc(self.dropout(combined))


# =============================================================================
# Data loading
# =============================================================================

def load_data() -> pd.DataFrame:
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)
    return df


def tokenize_texts(texts, tokenizer):
    print("Tokenizing reviews...")
    encoded = tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print(f"Tokenization complete. Shape: {encoded['input_ids'].shape}")
    return encoded["input_ids"], encoded["attention_mask"]


def build_meta_encoders(df):
    """Fit or reload drug/condition LabelEncoders (reuses train3.py encoders if available)."""
    drug_path = SAVED_MODEL_DIR / "drug_encoder.joblib"
    cond_path = SAVED_MODEL_DIR / "condition_encoder.joblib"

    if drug_path.exists() and cond_path.exists():
        print("Reusing drug/condition encoders from train3.py")
        drug_le = joblib.load(drug_path)
        cond_le = joblib.load(cond_path)
    else:
        print("Building drug/condition encoders from scratch")
        drug_le = LabelEncoder()
        cond_le = LabelEncoder()
        drug_le.fit(list(df["urlDrugName"].fillna("unknown").unique()) + ["unknown"])
        cond_le.fit(list(df["condition"].fillna("unknown").unique()) + ["unknown"])
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(drug_le, drug_path)
        joblib.dump(cond_le, cond_path)

    print(f"Drugs: {len(drug_le.classes_)}  |  Conditions: {len(cond_le.classes_)}")
    return drug_le, cond_le


# =============================================================================
# Training
# =============================================================================

def train_model(train_loader, val_loader, num_drugs, num_conditions,
                y_encoded_train, epochs=5, patience=2):
    model = DistilBERTMetadataClassifier(
        num_drugs=num_drugs,
        num_conditions=num_conditions,
    ).to(DEVICE)

    # Differential learning rates — DistilBERT layers at 2e-5, new layers at 1e-3
    distilbert_params = list(model.distilbert.parameters())
    new_layer_params  = (list(model.drug_embedding.parameters()) +
                         list(model.condition_embedding.parameters()) +
                         list(model.fc.parameters()))

    optimizer = torch.optim.AdamW([
        {"params": distilbert_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": new_layer_params,  "lr": 1e-3},
    ])

    # OneCycleLR: one max_lr per param group — 30% warmup then cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-5, 1e-3],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # Inverse-frequency class weights to address imbalance
    counts  = np.bincount(y_encoded_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    print(f"  Class weights: {dict(zip(range(len(counts)), weights.cpu().numpy().round(4)))}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_weights     = None

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for input_ids, attn_mask, drug, cond, y in train_bar:
            input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
            drug, cond, y        = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attn_mask, drug, cond), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]  ", leave=False)
        with torch.no_grad():
            for input_ids, attn_mask, drug, cond, y in val_bar:
                input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
                drug, cond, y        = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
                batch_loss = criterion(model(input_ids, attn_mask, drug, cond), y).item()
                val_loss  += batch_loss
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    print("DistilBERT training complete.")
    return model


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_model(model, input_ids, attention_masks, X_drug, X_cond, y_true, target_le, texts=None):
    model.eval()
    dataset = DistilBERTDataset(
        input_ids, attention_masks, X_drug, X_cond,
        np.zeros(len(X_drug), dtype=np.int64),
    )
    loader = DataLoader(dataset, batch_size=32)

    all_preds = []
    with torch.no_grad():
        for ids, mask, drug, cond, _ in loader:
            logits = model(ids.to(DEVICE), mask.to(DEVICE), drug.to(DEVICE), cond.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    y_pred  = target_le.inverse_transform(all_preds)
    y_true  = np.array(y_true)
    classes = target_le.classes_

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 13})
    plt.title("Confusion Matrix — DistilBERT + Metadata", fontsize=13)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    out = SAVED_MODEL_DIR / "confusion_matrix_train5.png"
    plt.savefig(out, dpi=150)
    print(f"Confusion matrix saved to {out}")
    plt.show()

    if texts is not None:
        print("\n--- Example Predictions ---")
        for text, actual, pred in zip(list(texts)[:5], list(y_true)[:5], list(y_pred)[:5]):
            print(f"  Text:      {str(text)[:80]}...")
            print(f"  Actual:    {actual}")
            print(f"  Predicted: {pred}\n")


# =============================================================================
# Save
# =============================================================================

def save_artifacts(model, target_le):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVED_MODEL_DIR / "model_distilbert_meta.pt")
    joblib.dump(target_le, SAVED_MODEL_DIR / "label_encoder_distilbert.joblib")
    print(f"Saved DistilBERT model to {SAVED_MODEL_DIR}")


# =============================================================================
# Main
# =============================================================================

def main():
    # 1. Load data
    df = load_data()

    # 2. Tokenize with DistilBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
    input_ids, attention_masks = tokenize_texts(df["benefitsReview"], tokenizer)

    # 3. Encode drug and condition
    drug_le, cond_le = build_meta_encoders(df)
    X_drug = safe_encode(drug_le, df["urlDrugName"].fillna("unknown"))
    X_cond = safe_encode(cond_le, df["condition"].fillna("unknown"))

    # 4. Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(df[TARGET_COL])

    # 5. Train/test split — hold out 20%
    indices = np.arange(len(y_encoded))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_encoded
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=0.1, random_state=42, stratify=y_encoded[idx_train]
    )

    train_dataset = DistilBERTDataset(
        input_ids[idx_train], attention_masks[idx_train],
        X_drug[idx_train], X_cond[idx_train], y_encoded[idx_train],
    )
    val_dataset = DistilBERTDataset(
        input_ids[idx_val], attention_masks[idx_val],
        X_drug[idx_val], X_cond[idx_val], y_encoded[idx_val],
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32)

    # 6. Train
    print(f"\n=== Training DistilBERT + Metadata on {len(idx_train):,} reviews ===")
    model = train_model(train_loader, val_loader,
                        num_drugs=len(drug_le.classes_),
                        num_conditions=len(cond_le.classes_),
                        y_encoded_train=y_encoded[idx_train])

    # 7. Evaluate on held-out test set
    print("\n=== Evaluation on held-out test set ===")
    y_test_labels = target_le.inverse_transform(y_encoded[idx_test])
    evaluate_model(
        model,
        input_ids[idx_test], attention_masks[idx_test],
        X_drug[idx_test], X_cond[idx_test],
        y_test_labels, target_le,
        texts=df["benefitsReview"].iloc[idx_test].values,
    )

    # 8. Save
    save_artifacts(model, target_le)
    print("Training complete!")


if __name__ == "__main__":
    main()
