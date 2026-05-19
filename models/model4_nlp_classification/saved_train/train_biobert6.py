#!/usr/bin/env python3
"""
Model 4: NLP Classification — BioBERT LoRA + OneCycleLR + Gradient Accumulation
=================================================================================
Copy of train_biobert5.py with two Phase 3 improvements borrowed from train_lstm0.py:

  1. OneCycleLR scheduler — linear warmup for 30% of steps then cosine decay.
     Tends to find better minima than linear warmup + linear decay.

  2. Gradient accumulation (ACCUM_STEPS=4) — effective batch size of 128
     (matching the LSTM's batch size), more stable gradient estimates.

Phase 2 is unchanged — reuses model_biobert_phase2.pt automatically.
Output: model_biobert6_lora_all_combos.pt

Phase 1 — BioBERT Pretraining (external, no code here)
    dmis-lab pretrained BioBERT on PubMed abstracts and PMC full-text articles.

Phase 2 — Domain Adaptation (reuses model_biobert_phase2.pt if it exists)
    Full fine-tuning on 192k patient reviews.

Phase 3 — LoRA + OneCycleLR + Gradient Accumulation on Full Dataset
    LoRA adapters on Q and V projections, frozen base weights, all combos.
    OneCycleLR cycles LR aggressively to escape local minima.
    Gradient accumulation simulates batch_size=128 on 32-sample batches.

Architecture:
    review text  → BioBERT mean-pool (768-dim) ─┐
    urlDrugName  → Embedding (32-dim) ───────────┤→ LayerNorm → proj → GELU → Dropout → FC
    condition    → Embedding (32-dim) ───────────┘
"""

import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (load_raw_data, clean_nlp_data, engineer_nlp_features,
                                          save_processed_data, load_processed_data)

PROCESSED_FILE  = "medication_feedback_processed.csv"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TARGET_COL      = "effectiveness_3class"
BIOBERT_MODEL   = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN         = 256
PHASE2_CKPT     = SAVED_MODEL_DIR / "model_biobert_phase2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Phase 2 knob ─────────────────────────────────────────────────────────────
PHASE2_FRAC = 1.00

# ── Phase 3 LoRA knobs ────────────────────────────────────────────────────────
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.1
LORA_LR       = 1e-4
HEAD_LR       = 5e-4
PHASE3_EPOCHS   = 8
PHASE3_PATIENCE = 3

# ── Gradient accumulation ─────────────────────────────────────────────────────
# Effective batch size = 32 * 4 = 128  (matches LSTM batch size)
ACCUM_STEPS = 4
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Dataset
# =============================================================================

class BioBERTDataset(Dataset):
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

class BioBERTMetadataClassifier(nn.Module):
    def __init__(self, num_drugs, num_conditions,
                 meta_embed_dim=32, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.bert                = AutoModel.from_pretrained(BIOBERT_MODEL)
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
        combined = F.gelu(self.proj(combined))
        return self.fc(self.dropout(combined))


# =============================================================================
# Data loading
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


def tokenize_texts(texts, tokenizer, label=""):
    print(f"Tokenizing {label}reviews...")
    encoded = tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print(f"Tokenization complete. Shape: {encoded['input_ids'].shape}")
    return encoded["input_ids"], encoded["attention_mask"]


def safe_encode(le: LabelEncoder, values) -> np.ndarray:
    known  = set(le.classes_)
    mapped = [v if v in known else "unknown" for v in values]
    return le.transform(mapped).astype(np.int64)


def build_meta_encoders(df):
    drug_path = SAVED_MODEL_DIR / "drug_encoder.joblib"
    cond_path = SAVED_MODEL_DIR / "condition_encoder.joblib"

    if drug_path.exists() and cond_path.exists():
        print("Reusing drug/condition encoders from prior run")
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
# Optimizer: LLRD (Phase 2 only)
# =============================================================================

def build_optimizer(model, base_lr=2e-5, decay=0.95, new_lr=1e-3):
    num_layers = model.bert.config.num_hidden_layers
    head_params = (
        list(model.drug_embedding.parameters()) +
        list(model.condition_embedding.parameters()) +
        list(model.norm.parameters()) +
        list(model.proj.parameters()) +
        list(model.fc.parameters())
    )
    groups = [{"params": head_params, "lr": new_lr}]
    groups.append({
        "params": list(model.bert.embeddings.parameters()),
        "lr": base_lr * (decay ** num_layers), "weight_decay": 0.01,
    })
    for i in range(num_layers):
        groups.append({
            "params": list(model.bert.encoder.layer[i].parameters()),
            "lr": base_lr * (decay ** (num_layers - 1 - i)), "weight_decay": 0.01,
        })
    groups.append({
        "params": list(model.bert.pooler.parameters()),
        "lr": base_lr, "weight_decay": 0.01,
    })
    return torch.optim.AdamW(groups)


# =============================================================================
# Training — Phase 2 (unchanged from biobert5)
# =============================================================================

def train_model(train_loader, val_loader, num_drugs, num_conditions,
                y_encoded_train, y_encoded_val,
                base_lr=2e-5, new_lr=1e-3, epochs=3, patience=2,
                pretrained_ckpt=None):
    model = BioBERTMetadataClassifier(num_drugs=num_drugs, num_conditions=num_conditions).to(DEVICE)
    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        state = torch.load(pretrained_ckpt, map_location=DEVICE)
        model_keys = set(model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in model_keys}
        model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint from {pretrained_ckpt}")

    optimizer    = build_optimizer(model, base_lr=base_lr, new_lr=new_lr)
    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    counts  = np.bincount(y_encoded_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_f1, patience_counter, best_weights = -1.0, 0, None
    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for ids, mask, drug, cond, y in bar:
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            drug, cond, y = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask, drug, cond), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval(); val_loss = 0.0; all_preds = []
        with torch.no_grad():
            for ids, mask, drug, cond, y in val_loader:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                drug, cond, y = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
                logits = model(ids, mask, drug, cond)
                val_loss += criterion(logits, y).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())

        val_f1 = f1_score(y_encoded_val, all_preds, average="weighted")
        print(f"Epoch {epoch+1}/{epochs} — train: {train_loss/len(train_loader):.4f}  "
              f"val: {val_loss/len(val_loader):.4f}  f1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    print(f"Phase 2 complete. Best val F1: {best_val_f1:.4f}")
    return model


# =============================================================================
# Training — Phase 3 (LoRA + OneCycleLR + gradient accumulation)
# =============================================================================

def train_model_lora(train_loader, val_loader, num_drugs, num_conditions,
                     y_encoded_train, y_encoded_val,
                     epochs=PHASE3_EPOCHS, patience=PHASE3_PATIENCE,
                     pretrained_ckpt=None):
    base_model = BioBERTMetadataClassifier(num_drugs=num_drugs, num_conditions=num_conditions).to(DEVICE)

    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        state = torch.load(pretrained_ckpt, map_location=DEVICE)
        model_keys = set(base_model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in model_keys}
        base_model.load_state_dict(state, strict=False)
        print(f"  Loaded Phase 2 checkpoint from {pretrained_ckpt}")

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    HEAD_NAMES = {"drug_embedding", "condition_embedding", "norm", "proj", "fc"}
    for name, param in model.named_parameters():
        if any(h in name for h in HEAD_NAMES):
            param.requires_grad = True

    model.print_trainable_parameters()

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in HEAD_NAMES) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR, "weight_decay": 0.01},
        {"params": head_params, "lr": HEAD_LR,  "weight_decay": 0.01},
    ])

    # OneCycleLR — one cycle per param group, 30% warmup then cosine decay
    steps_per_epoch = max(1, math.ceil(len(train_loader) / ACCUM_STEPS))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LORA_LR, HEAD_LR],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    counts  = np.bincount(y_encoded_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    print(f"  Class weights: {dict(zip(range(len(counts)), weights.cpu().numpy().round(4)))}")
    print(f"  Effective batch size: {32 * ACCUM_STEPS}  (accum steps: {ACCUM_STEPS})")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_f1, patience_counter, best_weights = -1.0, 0, None

    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        optimizer.zero_grad()
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for step, (ids, mask, drug, cond, y) in enumerate(bar):
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            drug, cond, y = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(ids, mask, drug, cond), y) / ACCUM_STEPS
            loss.backward()
            train_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            bar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

        model.eval(); val_loss = 0.0; all_preds = []
        with torch.no_grad():
            for ids, mask, drug, cond, y in val_loader:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                drug, cond, y = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
                logits = model(ids, mask, drug, cond)
                val_loss += criterion(logits, y).item()
                all_preds.extend(logits.argmax(1).cpu().numpy())

        val_f1 = f1_score(y_encoded_val, all_preds, average="weighted")
        print(f"Epoch {epoch+1}/{epochs} — train: {train_loss/len(train_loader):.4f}  "
              f"val: {val_loss/len(val_loader):.4f}  f1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(best_weights, SAVED_MODEL_DIR / "model_biobert6_best_ckpt.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val F1: {best_val_f1:.4f})")
                break

    model.load_state_dict(best_weights)
    print(f"LoRA training complete. Best val weighted-F1: {best_val_f1:.4f}")
    print("Merging LoRA adapters into base weights...")
    model = model.merge_and_unload()
    return model


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_model(model, input_ids, attention_masks, X_drug, X_cond, y_true, target_le):
    model.eval()
    dataset = BioBERTDataset(
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

    print("\n--- Classification Report (all combos) ---")
    print(classification_report(y_true, y_pred))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 13})
    plt.title("Confusion Matrix — BioBERT6 LoRA + OneCycleLR (all combos)", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    out = SAVED_MODEL_DIR / "confusion_matrix_biobert6_lora_all_combos.png"
    plt.savefig(out, dpi=150)
    print(f"Confusion matrix saved to {out}")
    plt.show()


# =============================================================================
# Save
# =============================================================================

def save_artifacts(model, target_le):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVED_MODEL_DIR / "model_biobert6_lora_all_combos.pt")
    joblib.dump(target_le, SAVED_MODEL_DIR / "label_encoder_biobert6_lora_all_combos.joblib")
    print(f"Saved model to {SAVED_MODEL_DIR / 'model_biobert6_lora_all_combos.pt'}")


# =============================================================================
# Main
# =============================================================================

def main():
    df_full  = load_data()
    drug_le, cond_le = build_meta_encoders(df_full)
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)

    # =========================================================================
    # Phase 2 — reuse checkpoint if available
    # =========================================================================
    if PHASE2_CKPT.exists():
        print(f"\n=== Phase 2: checkpoint found at {PHASE2_CKPT} — skipping ===")
    else:
        df_p2 = df_full.sample(frac=PHASE2_FRAC, random_state=42).reset_index(drop=True)
        print(f"\n=== Phase 2: domain adaptation on {len(df_p2):,} rows ===")

        ids_p2, mask_p2 = tokenize_texts(df_p2["benefitsReview"], tokenizer, label="Phase 2 ")
        X_drug_p2 = safe_encode(drug_le, df_p2["urlDrugName"].fillna("unknown"))
        X_cond_p2 = safe_encode(cond_le, df_p2["condition"].fillna("unknown"))

        target_le_p2 = LabelEncoder()
        y_p2 = target_le_p2.fit_transform(df_p2[TARGET_COL])
        idx_tr, idx_val = train_test_split(
            np.arange(len(y_p2)), test_size=0.1, random_state=42, stratify=y_p2)

        model_p2 = train_model(
            DataLoader(BioBERTDataset(ids_p2[idx_tr], mask_p2[idx_tr],
                                      X_drug_p2[idx_tr], X_cond_p2[idx_tr], y_p2[idx_tr]),
                       batch_size=32, shuffle=True),
            DataLoader(BioBERTDataset(ids_p2[idx_val], mask_p2[idx_val],
                                      X_drug_p2[idx_val], X_cond_p2[idx_val], y_p2[idx_val]),
                       batch_size=32),
            num_drugs=len(drug_le.classes_), num_conditions=len(cond_le.classes_),
            y_encoded_train=y_p2[idx_tr], y_encoded_val=y_p2[idx_val],
            base_lr=2e-5, new_lr=1e-3, epochs=3, patience=2,
        )
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model_p2.state_dict(), PHASE2_CKPT)
        print(f"Phase 2 checkpoint saved to {PHASE2_CKPT}")

    # =========================================================================
    # Phase 3 — LoRA + OneCycleLR + gradient accumulation, full dataset
    # =========================================================================
    print(f"\n=== Phase 3 (LoRA + OneCycleLR): full dataset ({len(df_full):,} rows) ===")
    print(f"    LoRA rank={LORA_R}, alpha={LORA_ALPHA}, accum_steps={ACCUM_STEPS}")

    ids_p3, mask_p3 = tokenize_texts(df_full["benefitsReview"], tokenizer, label="Phase 3 ")
    X_drug_p3 = safe_encode(drug_le, df_full["urlDrugName"].fillna("unknown"))
    X_cond_p3 = safe_encode(cond_le, df_full["condition"].fillna("unknown"))

    target_le = LabelEncoder()
    y_p3      = target_le.fit_transform(df_full[TARGET_COL])

    idx_train, idx_test = train_test_split(
        np.arange(len(y_p3)), test_size=0.1, random_state=42, stratify=y_p3)
    idx_train, idx_val = train_test_split(
        idx_train, test_size=0.1, random_state=42, stratify=y_p3[idx_train])

    print(f"  Train: {len(idx_train):,}  Val: {len(idx_val):,}  Test: {len(idx_test):,}")

    train_loader = DataLoader(
        BioBERTDataset(ids_p3[idx_train], mask_p3[idx_train],
                       X_drug_p3[idx_train], X_cond_p3[idx_train], y_p3[idx_train]),
        batch_size=32, shuffle=True,
    )
    val_loader = DataLoader(
        BioBERTDataset(ids_p3[idx_val], mask_p3[idx_val],
                       X_drug_p3[idx_val], X_cond_p3[idx_val], y_p3[idx_val]),
        batch_size=32,
    )

    model = train_model_lora(
        train_loader, val_loader,
        num_drugs=len(drug_le.classes_),
        num_conditions=len(cond_le.classes_),
        y_encoded_train=y_p3[idx_train],
        y_encoded_val=y_p3[idx_val],
        pretrained_ckpt=PHASE2_CKPT,
    )

    print(f"\n=== Evaluation on held-out test set ===")
    y_test_labels = target_le.inverse_transform(y_p3[idx_test])
    evaluate_model(
        model,
        ids_p3[idx_test], mask_p3[idx_test],
        X_drug_p3[idx_test], X_cond_p3[idx_test],
        y_test_labels, target_le,
    )

    save_artifacts(model, target_le)
    print("Training complete!")


if __name__ == "__main__":
    main()