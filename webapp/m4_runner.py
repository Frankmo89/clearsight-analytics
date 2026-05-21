#!/usr/bin/env python3
"""
M4 BioBERT inference runner — executes in an isolated subprocess.

Reads a JSON payload from stdin (keys: m4_dir, text, drug, cond),
runs the full BioBERT forward pass, then writes a JSON result to stdout.

No Streamlit imports. No TensorFlow imports. Safe for subprocess execution,
so PyTorch and TensorFlow never share the same process memory space.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import traceback


def main() -> None:
    payload    = json.loads(sys.stdin.read())
    m4_dir_str = payload["m4_dir"]
    text_notes = payload["text"]
    drug_name  = payload["drug"]
    condition  = payload["cond"]

    try:
        import numpy as np
        import joblib
        from pathlib import Path
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download

        M4_DIR  = Path(m4_dir_str)
        HF_REPO = "FrankAlonsoskyMolina/clearsight-analytics"

        def get_model_file(filename: str) -> Path:
            local_path = M4_DIR / filename
            if not local_path.exists():
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename,
                    local_dir=str(M4_DIR),
                )
            return local_path

        class BioBERTMetadataClassifier(nn.Module):
            def __init__(
                self,
                biobert_name: str,
                num_drugs: int,
                num_conditions: int,
                meta_embed_dim: int = 32,
                hidden_dim: int = 256,
                num_classes: int = 3,
                dropout: float = 0.3,
            ) -> None:
                super().__init__()
                self.bert              = AutoModel.from_pretrained(biobert_name)
                self.drug_embedding    = nn.Embedding(num_drugs, meta_embed_dim)
                self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
                combined_dim = 768 + meta_embed_dim * 2
                self.norm    = nn.LayerNorm(combined_dim)
                self.proj    = nn.Linear(combined_dim, hidden_dim)
                self.dropout = nn.Dropout(dropout)
                self.fc      = nn.Linear(hidden_dim, num_classes)

            @staticmethod
            def _mean_pool(
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                mask   = attention_mask.unsqueeze(-1).float()
                summed = (last_hidden_state * mask).sum(dim=1)
                return summed / mask.sum(dim=1).clamp(min=1)

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                drug_idx: torch.Tensor,
                cond_idx: torch.Tensor,
            ) -> torch.Tensor:
                outputs  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                text_out = self._mean_pool(outputs.last_hidden_state, attention_mask)
                drug_out = self.drug_embedding(drug_idx)
                cond_out = self.condition_embedding(cond_idx)
                combined = torch.cat([text_out, drug_out, cond_out], dim=1)
                combined = self.norm(combined)
                combined = torch.nn.functional.gelu(self.proj(combined))
                return self.fc(self.dropout(combined))

        # ── Load artefacts ──────────────────────────────────────────────────
        drug_le  = joblib.load(get_model_file("drug_encoder.joblib"))
        cond_le  = joblib.load(get_model_file("condition_encoder.joblib"))
        label_le = joblib.load(get_model_file("label_encoder_biobert_lora_all_combos.joblib"))

        biobert_name = "dmis-lab/biobert-base-cased-v1.2"
        tokenizer = AutoTokenizer.from_pretrained(biobert_name)
        model = BioBERTMetadataClassifier(
            biobert_name   = biobert_name,
            num_drugs      = len(drug_le.classes_),
            num_conditions = len(cond_le.classes_),
        )
        model.load_state_dict(
            torch.load(
                get_model_file("model_biobert_lora_all_combos.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()

        # ── Inference ───────────────────────────────────────────────────────
        def safe_encode(le, val: str) -> int:
            return le.transform([val if val in le.classes_ else "unknown"])[0]

        drug_idx = torch.tensor([safe_encode(drug_le, drug_name)], dtype=torch.long)
        cond_idx = torch.tensor([safe_encode(cond_le, condition)], dtype=torch.long)

        enc = tokenizer(
            str(text_notes),
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits   = model(enc["input_ids"], enc["attention_mask"], drug_idx, cond_idx)
            probs    = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))

        label      = label_le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        explanation_map = {
            "Ineffective":        "Critical Interpretation: Linguistic markers suggest severe symptoms, treatment failure, or a potential emergency. Immediate review advised.",
            "Somewhat Effective": "Elevated Interpretation: Linguistic markers indicate lingering symptoms or an incomplete response to current treatment context.",
            "Highly Effective":   "Stable Interpretation: Linguistic markers indicate a positive response to treatment and stable patient condition.",
        }
        css_map = {
            "Ineffective":        "risk-high",
            "Somewhat Effective": "risk-medium",
            "Highly Effective":   "risk-low",
        }
        display_map = {
            "Ineffective":        "CRITICAL",
            "Somewhat Effective": "ELEVATED",
            "Highly Effective":   "STABLE",
        }

        print(json.dumps({
            "success":     True,
            "label":       f"{display_map.get(label, label.upper())} RISK SENTIMENT",
            "confidence":  confidence,
            "css":         css_map.get(label, "risk-low"),
            "explanation": explanation_map.get(label, "Interpretation unavailable."),
        }))
        sys.exit(0)

    except Exception as exc:
        print(json.dumps({
            "success":   False,
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
