#!/usr/bin/env python3
"""
Model 4: NLP Classification — BioBERT LoRA trained on ALL drug/condition combos
=================================================================================
Phase 2: Same full fine-tuning on 192k rows as train_biobert1/2.py.
    Reuses model_biobert_phase2.pt if it exists — no re-training needed.

Phase 3: Iterates over every drug/condition combination with >= MIN_ROWS reviews
    and trains a separate LoRA-adapted BioBERT model for each one.
    Each model is saved as model_biobert_lora_{drug}_{condition}.pt.
    Already-trained combinations are skipped automatically (resume-safe).

The full dataset is tokenized once upfront to avoid redundant work across
the hundreds of combination loops.

A summary CSV (lora_combo_results.csv) is written after every combo so the
run can be interrupted and resumed without losing results.

Knobs:
    MIN_ROWS     — minimum reviews for a combo to be trained (default 200)
    LORA_R       — LoRA adapter rank
    LORA_ALPHA   — LoRA scaling (keep at 2x rank)
    PHASE2_FRAC  — fraction of 192k used if Phase 2 needs to run
"""

import re
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no blocking plt.show()
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import load_raw_data, clean_nlp_data, engineer_nlp_features

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
LORA_MODEL_DIR  = SAVED_MODEL_DIR / "lora_combos"
TARGET_COL      = "effectiveness_3class"
BIOBERT_MODEL   = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN         = 256
PHASE2_CKPT     = SAVED_MODEL_DIR / "model_biobert_phase2.pt"
RESULTS_CSV     = LORA_MODEL_DIR / "lora_combo_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Combo filter ──────────────────────────────────────────────────────────────
# Only train combos with at least this many reviews.
# 200 → ~144 training rows after splits; lower = noisier models.
MIN_ROWS = 200

# ── Phase 2 knob ─────────────────────────────────────────────────────────────
PHASE2_FRAC = 1.00

# ── LoRA knobs ────────────────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def make_slug(drug: str, condition: str) -> str:
    def clean(s):
        return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")
    return f"{clean(drug)}_{clean(condition)}"


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
# Data helpers
# =============================================================================

def load_data() -> pd.DataFrame:
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_nlp_data(df)
    df = engineer_nlp_features(df)
    return df


def tokenize_texts(texts, tokenizer, label=""):
    print(f"Tokenizing {label}...")
    encoded = tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print(f"  Done. Shape: {encoded['input_ids'].shape}")
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
# Optimizer (Phase 2 only)
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
# Training — Phase 2
# =============================================================================

def train_model(train_loader, val_loader, num_drugs, num_conditions,
                y_encoded_train, y_encoded_val,
                base_lr=2e-5, new_lr=1e-3, epochs=8, patience=3,
                pretrained_ckpt=None):
    model = BioBERTMetadataClassifier(num_drugs=num_drugs, num_conditions=num_conditions).to(DEVICE)
    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=DEVICE))
        print(f"  Loaded checkpoint from {pretrained_ckpt}")

    optimizer    = build_optimizer(model, base_lr=base_lr, new_lr=new_lr)
    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    counts   = np.bincount(y_encoded_train)
    weights  = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_f1, patience_counter, best_weights = -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss/len(train_loader):.4f}  "
              f"val_loss: {val_loss/len(val_loader):.4f}  val_f1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
# Training — Phase 3 LoRA
# =============================================================================

def train_model_lora(train_loader, val_loader, num_drugs, num_conditions,
                     y_encoded_train, y_encoded_val,
                     lora_lr=1e-4, head_lr=5e-4,
                     epochs=10, patience=4,
                     pretrained_ckpt=None):
    base_model = BioBERTMetadataClassifier(num_drugs=num_drugs, num_conditions=num_conditions).to(DEVICE)

    if pretrained_ckpt is not None and Path(pretrained_ckpt).exists():
        state = torch.load(pretrained_ckpt, map_location=DEVICE)
        model_keys = set(base_model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in model_keys}
        base_model.load_state_dict(state, strict=False)

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

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in HEAD_NAMES) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lora_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
    ])

    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    counts   = np.bincount(y_encoded_train)
    weights  = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_f1, patience_counter, best_weights = -1.0, 0, None
    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} [train]", leave=False)
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
        print(f"  Epoch {epoch+1}/{epochs} — train: {train_loss/len(train_loader):.4f}  "
              f"val: {val_loss/len(val_loader):.4f}  f1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val F1: {best_val_f1:.4f})")
                break

    model.load_state_dict(best_weights)
    model = model.merge_and_unload()
    return model, best_val_f1


# =============================================================================
# Per-combo evaluate + save
# =============================================================================

def evaluate_and_save(model, ids_test, masks_test, X_drug_test, X_cond_test,
                      y_test_enc, target_le, drug, condition):
    model.eval()
    dataset = BioBERTDataset(ids_test, masks_test, X_drug_test, X_cond_test,
                             np.zeros(len(X_drug_test), dtype=np.int64))
    loader  = DataLoader(dataset, batch_size=32)

    all_preds = []
    with torch.no_grad():
        for ids, mask, d, c, _ in loader:
            logits = model(ids.to(DEVICE), mask.to(DEVICE), d.to(DEVICE), c.to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().numpy())

    y_pred   = target_le.inverse_transform(all_preds)
    y_true   = target_le.inverse_transform(y_test_enc)
    classes  = target_le.classes_
    test_f1  = f1_score(y_true, y_pred, average="weighted")

    print(f"  Test weighted F1: {test_f1:.4f}")
    print(classification_report(y_true, y_pred))

    slug = make_slug(drug, condition)
    cm   = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={"size": 13})
    ax.set_title(f"BioBERT LoRA — {drug} / {condition}", fontsize=11)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    cm_path = LORA_MODEL_DIR / f"cm_{slug}.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    LORA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LORA_MODEL_DIR / f"model_{slug}.pt")
    joblib.dump(target_le, LORA_MODEL_DIR / f"label_encoder_{slug}.joblib")

    return test_f1


# =============================================================================
# Main
# =============================================================================

def main():
    LORA_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_full  = load_data()
    drug_le, cond_le = build_meta_encoders(df_full)
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)

    # =========================================================================
    # Phase 2 — reuse checkpoint from train_biobert1/2 if available
    # =========================================================================
    if PHASE2_CKPT.exists():
        print(f"\n=== Phase 2: checkpoint found — skipping ===")
    else:
        df_p2 = df_full.sample(frac=PHASE2_FRAC, random_state=42).reset_index(drop=True)
        print(f"\n=== Phase 2: {len(df_p2):,} rows ({PHASE2_FRAC:.0%}) ===")
        ids_p2, mask_p2 = tokenize_texts(df_p2["benefitsReview"], tokenizer, "Phase 2 reviews")
        X_drug_p2 = safe_encode(drug_le, df_p2["urlDrugName"].fillna("unknown"))
        X_cond_p2 = safe_encode(cond_le, df_p2["condition"].fillna("unknown"))
        target_le_p2 = LabelEncoder()
        y_p2 = target_le_p2.fit_transform(df_p2[TARGET_COL])
        idx_tr, idx_val = train_test_split(np.arange(len(y_p2)), test_size=0.1,
                                           random_state=42, stratify=y_p2)
        tr_loader = DataLoader(
            BioBERTDataset(ids_p2[idx_tr], mask_p2[idx_tr],
                           X_drug_p2[idx_tr], X_cond_p2[idx_tr], y_p2[idx_tr]),
            batch_size=32, shuffle=True)
        va_loader = DataLoader(
            BioBERTDataset(ids_p2[idx_val], mask_p2[idx_val],
                           X_drug_p2[idx_val], X_cond_p2[idx_val], y_p2[idx_val]),
            batch_size=32)
        model_p2 = train_model(tr_loader, va_loader,
                               len(drug_le.classes_), len(cond_le.classes_),
                               y_p2[idx_tr], y_p2[idx_val],
                               base_lr=2e-5, new_lr=1e-3, epochs=3, patience=2)
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model_p2.state_dict(), PHASE2_CKPT)
        print(f"Phase 2 checkpoint saved to {PHASE2_CKPT}")

    # =========================================================================
    # Tokenize the entire dataset once — reused across all combo loops
    # =========================================================================
    print("\nTokenizing full dataset (one-time, reused for all combos)...")
    all_ids, all_masks = tokenize_texts(df_full["benefitsReview"], tokenizer, "all reviews")
    X_drug_all = safe_encode(drug_le, df_full["urlDrugName"].fillna("unknown"))
    X_cond_all = safe_encode(cond_le, df_full["condition"].fillna("unknown"))

    # Global label encoder — all 3 classes always present
    global_target_le = LabelEncoder().fit(df_full[TARGET_COL])
    y_all = global_target_le.transform(df_full[TARGET_COL])

    # =========================================================================
    # Build combo list
    # =========================================================================
    combos = (
        df_full
        .assign(drug_norm=df_full["urlDrugName"].str.lower().str.strip(),
                cond_norm=df_full["condition"].str.lower().str.strip())
        .groupby(["drug_norm", "cond_norm"])
        .size()
        .reset_index(name="count")
        .query(f"count >= {MIN_ROWS}")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n{len(combos)} drug/condition combos with >= {MIN_ROWS} reviews")

    # Load existing results if resuming
    if RESULTS_CSV.exists():
        results_df = pd.read_csv(RESULTS_CSV)
        done_slugs = set(results_df["slug"].tolist())
        print(f"Resuming — {len(done_slugs)} combos already complete")
    else:
        results_df = pd.DataFrame()
        done_slugs = set()

    # =========================================================================
    # Phase 3 loop — one LoRA model per combo
    # =========================================================================
    for i, row in combos.iterrows():
        drug      = row["drug_norm"]
        condition = row["cond_norm"]
        n_rows    = row["count"]
        slug      = make_slug(drug, condition)

        print(f"\n[{i+1}/{len(combos)}] {drug!r} / {condition!r}  ({n_rows:,} rows)")

        if slug in done_slugs:
            print("  Already trained — skipping")
            continue

        # Index into pre-tokenized tensors
        combo_mask = (
            df_full["urlDrugName"].str.lower().str.strip().eq(drug) &
            df_full["condition"].str.lower().str.strip().eq(condition)
        ).values
        combo_idx = np.where(combo_mask)[0]

        ids_sub   = all_ids[combo_idx]
        masks_sub = all_masks[combo_idx]
        X_drug_sub = X_drug_all[combo_idx]
        X_cond_sub = X_cond_all[combo_idx]
        y_sub      = y_all[combo_idx]

        try:
            idx_train, idx_test = train_test_split(
                np.arange(len(y_sub)), test_size=0.2, random_state=42, stratify=y_sub)
            idx_train, idx_val = train_test_split(
                idx_train, test_size=0.1, random_state=42, stratify=y_sub[idx_train])
        except ValueError as e:
            print(f"  Skipping — split failed ({e})")
            continue

        tr_loader = DataLoader(
            BioBERTDataset(ids_sub[idx_train], masks_sub[idx_train],
                           X_drug_sub[idx_train], X_cond_sub[idx_train], y_sub[idx_train]),
            batch_size=32, shuffle=True)
        va_loader = DataLoader(
            BioBERTDataset(ids_sub[idx_val], masks_sub[idx_val],
                           X_drug_sub[idx_val], X_cond_sub[idx_val], y_sub[idx_val]),
            batch_size=32)

        try:
            model, best_val_f1 = train_model_lora(
                tr_loader, va_loader,
                num_drugs=len(drug_le.classes_),
                num_conditions=len(cond_le.classes_),
                y_encoded_train=y_sub[idx_train],
                y_encoded_val=y_sub[idx_val],
                lora_lr=1e-4, head_lr=5e-4,
                epochs=10, patience=4,
                pretrained_ckpt=PHASE2_CKPT,
            )
        except Exception as e:
            print(f"  Training failed — {e}")
            continue

        test_f1 = evaluate_and_save(
            model,
            ids_sub[idx_test], masks_sub[idx_test],
            X_drug_sub[idx_test], X_cond_sub[idx_test],
            y_sub[idx_test], global_target_le,
            drug, condition,
        )

        # Append result and flush to CSV immediately
        new_row = pd.DataFrame([{
            "slug": slug, "drug": drug, "condition": condition,
            "n_rows": n_rows, "val_f1": round(best_val_f1, 4),
            "test_f1": round(test_f1, 4),
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(RESULTS_CSV, index=False)
        done_slugs.add(slug)

        # Free GPU memory between combos
        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Final summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"All combos complete. Results saved to {RESULTS_CSV}")
    if not results_df.empty:
        print(f"Combos trained:  {len(results_df)}")
        print(f"Mean test F1:    {results_df['test_f1'].mean():.4f}")
        print(f"Median test F1:  {results_df['test_f1'].median():.4f}")
        print(f"Best combo:      {results_df.loc[results_df['test_f1'].idxmax(), 'slug']}  "
              f"({results_df['test_f1'].max():.4f})")
        print(f"Worst combo:     {results_df.loc[results_df['test_f1'].idxmin(), 'slug']}  "
              f"({results_df['test_f1'].min():.4f})")
        print("\nTop 10 by test F1:")
        print(results_df.nlargest(10, "test_f1")[["drug", "condition", "n_rows", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()