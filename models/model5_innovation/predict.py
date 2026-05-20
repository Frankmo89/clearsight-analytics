#!/usr/bin/env python3
"""
Model 5: Innovation — LOS Prediction Script
=============================================
Loads trained model and predicts hospital stay duration tiers.

Usage: python models/model5_innovation/predict.py
Output: test_data/model5_results.csv
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import load_and_clean, engineer_features

# Paths
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model5_results.csv"

HF_REPO      = "FrankAlonsoskyMolina/clearsight-analytics"
HF_SUBFOLDER = "model5_innovation/saved_model"

LOS_LABELS = ['short_stay', 'medium_stay', 'extended_stay']


def ensure_model_files():
    """Download all saved_model files from HuggingFace if any are missing."""
    if not any(SAVED_MODEL_DIR.glob("*.joblib")):
        print("Model files not found locally — downloading from HuggingFace...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=HF_REPO,
                allow_patterns=[f"{HF_SUBFOLDER}/*"],
                local_dir=str(PROJECT_ROOT / "models"),
            )
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Could not download model files from HuggingFace ({HF_REPO}). Error: {e}"
            )


def load_model():
    """Load trained model and artifacts."""
    ensure_model_files()
    model = joblib.load(SAVED_MODEL_DIR / "model.joblib")
    preprocessing_state = joblib.load(SAVED_MODEL_DIR / "preprocessing_state.joblib")
    feature_names = joblib.load(SAVED_MODEL_DIR / "feature_names.joblib")
    metric_value = joblib.load(SAVED_MODEL_DIR / "metric_value.joblib")
    return model, preprocessing_state, feature_names, metric_value


def find_test_file():
    """Find the test data CSV in test_data/, falling back to data/raw/."""
    csv_files = list(TEST_DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if "results" not in f.name.lower()]
    if not csv_files:
        fallback = PROJECT_ROOT / "data" / "raw" / "patient_encounters_2023.csv"
        if fallback.exists():
            print(f"No file in test_data/ — using fallback: {fallback}")
            return fallback
        raise FileNotFoundError(
            f"No test data CSV found in {TEST_DATA_DIR}/ or {fallback.parent}/"
        )
    return csv_files[0]


def main():
    # 1. Load model
    print("Loading model...")
    model, preprocessing_state, feature_names, metric_value = load_model()

    # 2. Find and load test data
    test_file = find_test_file()
    print(f"Test file found: {test_file}")

    # 3. Preprocess: load, clean, engineer features
    df = load_and_clean(str(test_file))

    # Capture IDs after load_and_clean (sorted by encounter_id)
    if 'encounter_id' in df.columns:
        ids = df['encounter_id'].values.copy()
    else:
        ids = np.arange(len(df))

    df, _ = engineer_features(df, preprocessing_state=preprocessing_state)

    # Drop leaky columns (same as training)
    leak_cols = ['time_in_hospital', 'los_tier', 'diagnoses_per_day',
                 'readmission_binary']
    X_test = df.drop(columns=[c for c in leak_cols if c in df.columns])

    # Align columns to training order
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # 4. Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)

    # Map numeric predictions to tier labels
    predicted_labels = [LOS_LABELS[p] for p in y_pred]

    # 5. Save results — MUST match model5 template
    results = pd.DataFrame({
        "id": ids,
        "prediction": predicted_labels,
        "confidence": confidence.round(4),
        "metric_name": "weighted_f1",
        "metric_value": round(metric_value, 4),
    })
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")
    print(f"Total predictions: {len(results)}")
    print(f"Distribution: {dict(zip(*np.unique(predicted_labels, return_counts=True)))}")


if __name__ == "__main__":
    main()