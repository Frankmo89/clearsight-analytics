#!/usr/bin/env python3
"""
Model 4: NLP Classification — BioBERT 3-Phase Training with LoRA on Phase 3
=============================================================================
Phase 2 (this file): Same as train_biobert1.py — full fine-tuning on 192k rows.
    Reuses the model_biobert_phase2.pt checkpoint if it already exists.

Phase 3 (this file): LoRA fine-tuning on target drug/condition subset.
    Base BioBERT weights are FROZEN. Only LoRA adapter matrices (~0.3% of params)
    + classification head are trained. This prevents catastrophic forgetting
    completely and dramatically reduces overfitting on the small subset.

After Phase 3, LoRA adapters are merged back into the base weights so the
saved model is a plain BioBERTMetadataClassifier — no peft dependency at
inference time.

LoRA knobs:
    LORA_R       — rank of adapter matrices (higher = more capacity, more params)
    LORA_ALPHA   — scaling factor (typically 2x rank)
    LORA_DROPOUT — dropout on adapter paths

Architecture (same as train_biobert0/1.py):
    review text  → BioBERT mean-pool (768-dim) ─┐
    urlDrugName  → Embedding (32-dim) ───────────┤→ LayerNorm → proj → GELU → Dropout → FC
    condition    → Embedding (32-dim) ───────────┘
"""

import sys
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

from pipelines.data_pipeline import load_raw_data, clean_nlp_data, engineer_nlp_features

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TARGET_COL      = "effectiveness_3class"
BIOBERT_MODEL   = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN         = 256
PHASE2_CKPT     = SAVED_MODEL_DIR / "model_biobert_phase2.pt"   # shared with train_biobert1.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Phase 2 knob ─────────────────────────────────────────────────────────────
PHASE2_FRAC = 1.00   # fraction of 192k used for Phase 2 (1.0 = full dataset)

# ── Phase 3 LoRA knobs ───────────────────────────────────────────────────────
LORA_R       = 16    # adapter rank — 8 is lighter, 32 is more expressive
LORA_ALPHA   = 32    # scaling = alpha / r; keep at 2x r
LORA_DROPOUT = 0.1

# ── Phase 3 target ───────────────────────────────────────────────────────────
FILTER_DRUG      = "etonogestrel"
FILTER_CONDITION = "birth control"
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
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)
    return df


def load_filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["urlDrugName"].str.lower().str.strip().eq(FILTER_DRUG) &
        df["condition"].str.lower().str.strip().eq(FILTER_CONDITION)
    )
    filtered = df[mask].reset_index(drop=True)
    print(f"\nFiltered to {FILTER_DRUG!r} / {FILTER_CONDITION!r}: {len(filtered):,} rows")
    print(filtered[TARGET_COL].value_counts())
    return filtered


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
# Optimizer: layer-wise learning rate decay (used for Phase 2 only)
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
        "lr": base_lr * (decay ** num_layers),
        "weight_decay": 0.01,
    })

    for i in range(num_layers):
        groups.append({
            "params": list(model.bert.encoder.layer[i].parameters()),
            "lr": base_lr * (decay ** (num_layers - 1 - i)),
            "weight_decay": 0.01,
        })

    groups.append({
        "params": list(model.bert.pooler.parameters()),
        "lr": base_lr,
        "weight_decay": 0.01,
    })

    return torch.optim.AdamW(groups)


# =============================================================================
# Training — Phase 2 (full fine-tuning with LLRD)
# =============================================================================

def train_model(train_loader, val_loader, num_drugs, num_conditions,
                y_encoded_train, y_encoded_val,
                base_lr=2e-5, new_lr=1e-3,
                epochs=8, patience=3,
                pretrained_ckpt=None):
    model = BioBERTMetadataClassifier(
        num_drugs=num_drugs,
        num_conditions=num_conditions,
    ).to(DEVICE)

    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=DEVICE))
        print(f"  Loaded checkpoint from {pretrained_ckpt}")

    optimizer    = build_optimizer(model, base_lr=base_lr, new_lr=new_lr)
    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    counts  = np.bincount(y_encoded_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    print(f"  Class weights: {dict(zip(range(len(counts)), weights.cpu().numpy().round(4)))}")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_f1      = -1.0
    patience_counter = 0
    best_weights     = None

    for epoch in range(epochs):
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

        model.eval()
        val_loss  = 0.0
        all_preds = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]  ", leave=False)
        with torch.no_grad():
            for input_ids, attn_mask, drug, cond, y in val_bar:
                input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
                drug, cond, y        = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
                logits     = model(input_ids, attn_mask, drug, cond)
                batch_loss = criterion(logits, y).item()
                val_loss  += batch_loss
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_f1 = f1_score(y_encoded_val, all_preds, average="weighted")
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  val_f1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val F1: {best_val_f1:.4f})")
                break

    model.load_state_dict(best_weights)
    print(f"Training complete. Best val weighted-F1: {best_val_f1:.4f}")
    return model


# =============================================================================
# Training — Phase 3 (LoRA fine-tuning)
# =============================================================================

def train_model_lora(train_loader, val_loader, num_drugs, num_conditions,
                     y_encoded_train, y_encoded_val,
                     lora_lr=1e-4, head_lr=5e-4,
                     epochs=10, patience=4,
                     pretrained_ckpt=None):
    """
    Applies LoRA adapters to BioBERT's Q and V attention projections.
    Base BioBERT weights are frozen; only LoRA params + classification head train.
    After training, adapters are merged back into the base weights so the
    returned model is a plain BioBERTMetadataClassifier (no peft at inference).
    """
    base_model = BioBERTMetadataClassifier(
        num_drugs=num_drugs,
        num_conditions=num_conditions,
    ).to(DEVICE)

    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        state = torch.load(pretrained_ckpt, map_location=DEVICE)
        # Strip any UNEXPECTED keys (e.g. cls.* from BioBERT pretraining head)
        model_keys = set(base_model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in model_keys}
        base_model.load_state_dict(state, strict=False)
        print(f"  Loaded Phase 2 checkpoint from {pretrained_ckpt}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],   # Q and V projections in every BERT layer
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    # Unfreeze classification head and meta embeddings — peft freezes everything
    # except the LoRA adapters by default
    HEAD_NAMES = {"drug_embedding", "condition_embedding", "norm", "proj", "fc"}
    for name, param in model.named_parameters():
        if any(h in name for h in HEAD_NAMES):
            param.requires_grad = True

    model.print_trainable_parameters()

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in HEAD_NAMES) and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lora_lr,  "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr,  "weight_decay": 0.01},
    ])

    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    counts  = np.bincount(y_encoded_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum()
    print(f"  Class weights: {dict(zip(range(len(counts)), weights.cpu().numpy().round(4)))}")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_f1      = -1.0
    patience_counter = 0
    best_weights     = None

    for epoch in range(epochs):
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

        model.eval()
        val_loss  = 0.0
        all_preds = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]  ", leave=False)
        with torch.no_grad():
            for input_ids, attn_mask, drug, cond, y in val_bar:
                input_ids, attn_mask = input_ids.to(DEVICE), attn_mask.to(DEVICE)
                drug, cond, y        = drug.to(DEVICE), cond.to(DEVICE), y.to(DEVICE)
                logits     = model(input_ids, attn_mask, drug, cond)
                batch_loss = criterion(logits, y).item()
                val_loss  += batch_loss
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_bar.set_postfix(loss=f"{batch_loss:.4f}")

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_f1 = f1_score(y_encoded_val, all_preds, average="weighted")
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  val_f1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val F1: {best_val_f1:.4f})")
                break

    model.load_state_dict(best_weights)
    print(f"LoRA training complete. Best val weighted-F1: {best_val_f1:.4f}")

    # Merge LoRA adapters back into base weights — inference needs no peft
    print("Merging LoRA adapters into base weights...")
    model = model.merge_and_unload()
    return model


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_model(model, input_ids, attention_masks, X_drug, X_cond,
                   y_true, target_le, texts=None):
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

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    slug = f"{FILTER_DRUG.replace(' ', '_')}_{FILTER_CONDITION.replace(' ', '_')}"
    cm   = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 13})
    plt.title(f"Confusion Matrix — BioBERT LoRA 3-phase ({FILTER_DRUG} / {FILTER_CONDITION})",
              fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    out = SAVED_MODEL_DIR / f"confusion_matrix_biobert_lora_{slug}.png"
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
    slug = f"{FILTER_DRUG.replace(' ', '_')}_{FILTER_CONDITION.replace(' ', '_')}"
    torch.save(model.state_dict(), SAVED_MODEL_DIR / f"model_biobert_lora_{slug}.pt")
    joblib.dump(target_le, SAVED_MODEL_DIR / f"label_encoder_biobert_lora_{slug}.joblib")
    print(f"Saved LoRA 3-phase BioBERT model to {SAVED_MODEL_DIR}")


# =============================================================================
# Main
# =============================================================================

def main():
    # 1. Load full dataset
    df_full = load_data()
    drug_le, cond_le = build_meta_encoders(df_full)
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)

    # =========================================================================
    # Phase 2 — full fine-tuning on full corpus (reuses biobert1 checkpoint)
    # =========================================================================
    if PHASE2_CKPT.exists():
        print(f"\n=== Phase 2: checkpoint found at {PHASE2_CKPT} — skipping ===")
    else:
        df_p2 = df_full.sample(frac=PHASE2_FRAC, random_state=42).reset_index(drop=True)
        print(f"\n=== Phase 2: domain adaptation on {len(df_p2):,} rows "
              f"({PHASE2_FRAC:.0%} of full dataset) ===")

        ids_p2, mask_p2 = tokenize_texts(df_p2["benefitsReview"], tokenizer, label="Phase 2 ")
        X_drug_p2 = safe_encode(drug_le, df_p2["urlDrugName"].fillna("unknown"))
        X_cond_p2 = safe_encode(cond_le, df_p2["condition"].fillna("unknown"))

        target_le_p2 = LabelEncoder()
        y_p2         = target_le_p2.fit_transform(df_p2[TARGET_COL])

        indices = np.arange(len(y_p2))
        idx_tr, idx_val = train_test_split(
            indices, test_size=0.1, random_state=42, stratify=y_p2
        )

        train_loader = DataLoader(
            BioBERTDataset(ids_p2[idx_tr], mask_p2[idx_tr],
                           X_drug_p2[idx_tr], X_cond_p2[idx_tr], y_p2[idx_tr]),
            batch_size=32, shuffle=True,
        )
        val_loader = DataLoader(
            BioBERTDataset(ids_p2[idx_val], mask_p2[idx_val],
                           X_drug_p2[idx_val], X_cond_p2[idx_val], y_p2[idx_val]),
            batch_size=32,
        )

        model_p2 = train_model(
            train_loader, val_loader,
            num_drugs=len(drug_le.classes_),
            num_conditions=len(cond_le.classes_),
            y_encoded_train=y_p2[idx_tr],
            y_encoded_val=y_p2[idx_val],
            base_lr=2e-5, new_lr=1e-3,
            epochs=3, patience=2,
        )

        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model_p2.state_dict(), PHASE2_CKPT)
        print(f"Phase 2 checkpoint saved to {PHASE2_CKPT}")

    # =========================================================================
    # Phase 3 — LoRA fine-tuning on target drug/condition subset
    # =========================================================================
    df_sub = load_filtered_data(df_full)
    print(f"\n=== Phase 3 (LoRA): fine-tuning on {FILTER_DRUG!r} / {FILTER_CONDITION!r} ===")
    print(f"    LoRA rank={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

    ids_p3, mask_p3 = tokenize_texts(df_sub["benefitsReview"], tokenizer, label="Phase 3 ")
    X_drug_p3 = safe_encode(drug_le, df_sub["urlDrugName"].fillna("unknown"))
    X_cond_p3 = safe_encode(cond_le, df_sub["condition"].fillna("unknown"))

    target_le = LabelEncoder()
    y_p3      = target_le.fit_transform(df_sub[TARGET_COL])

    indices = np.arange(len(y_p3))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_p3
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=0.1, random_state=42, stratify=y_p3[idx_train]
    )

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
        lora_lr=1e-4, head_lr=5e-4,
        epochs=10, patience=4,
        pretrained_ckpt=PHASE2_CKPT,
    )

    # Evaluate on held-out test split
    print(f"\n=== Evaluation on held-out {FILTER_DRUG!r} / {FILTER_CONDITION!r} reviews ===")
    y_test_labels = target_le.inverse_transform(y_p3[idx_test])
    evaluate_model(
        model,
        ids_p3[idx_test], mask_p3[idx_test],
        X_drug_p3[idx_test], X_cond_p3[idx_test],
        y_test_labels, target_le,
        texts=df_sub["benefitsReview"].iloc[idx_test].values,
    )

    save_artifacts(model, target_le)
    print("Training complete!")


if __name__ == "__main__":
    main()