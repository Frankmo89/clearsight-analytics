#!/usr/bin/env python3
"""
Model 1: Traditional ML — Prediction Script
=============================================
Loads trained XGBoost model and generates predictions on test data.

Usage: python models/model1_traditional_ml/predict.py
Output: test_data/model1_results.csv
"""
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import prepare_test_data

# Paths
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model1_results.csv"

HF_REPO      = "FrankAlonsoskyMolina/clearsight-analytics"
HF_SUBFOLDER = "model1_traditional_ml/saved_model"


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
    """Load trained model, preprocessing state, feature names, and optimal threshold."""
    ensure_model_files()
    model = joblib.load(SAVED_MODEL_DIR / "model.joblib")
    preprocessing_state = joblib.load(SAVED_MODEL_DIR / "preprocessing_state.joblib")
    feature_names = joblib.load(SAVED_MODEL_DIR / "feature_names.joblib")

    threshold_path = SAVED_MODEL_DIR / "optimal_threshold.joblib"
    optimal_threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5

    return model, preprocessing_state, feature_names, optimal_threshold


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

    # Use the first non-results CSV found
    return csv_files[0]


def predict(model, X_test, feature_names, threshold=0.5):
    """Generate predictions using cost-optimized threshold.

    Predictions use the optimal threshold from training (minimizes clinical
    cost: FN = $15K, FP = $500), not the default 0.5.
    """
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Confidence = how sure the model is (max of both probabilities)
    confidence = np.maximum(y_proba, 1 - y_proba)

    return y_pred, y_proba, confidence


def main():
    # 1. Load model and preprocessing artifacts
    print("Loading model...")
    model, preprocessing_state, feature_names, optimal_threshold = load_model()
    print(f"Using cost-optimized threshold: {optimal_threshold:.2f}")

    # 2. Find and load test data
    test_file = find_test_file()
    print(f"Test file found: {test_file}")



    # 3. Preprocess test data using same pipeline as training
    #    prepare_test_data returns aligned IDs (safe after sort + filter)
    X_test, ids = prepare_test_data(str(test_file), preprocessing_state)

    # 4. Generate predictions (using cost-optimized threshold)
    y_pred, y_proba, confidence = predict(model, X_test, feature_names, threshold=optimal_threshold)

    # 5. Save results — MUST match output template exactly
    results = pd.DataFrame({
        "id": ids,
        "prediction": y_pred,
        "probability": y_proba.round(4),
        "confidence": confidence.round(4),
    })
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")
    print(f"Total predictions: {len(results)}")
    print(f"Prediction distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")


if __name__ == "__main__":
    main()