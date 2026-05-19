#!/usr/bin/env python3
"""
Model 2: Deep Learning — Prediction Script
============================================
Loads trained DNN model and generates predictions on test data.

Usage: python models/model2_deep_learning/predict.py
Output: test_data/model2_results.csv
"""
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

# Add project root to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import prepare_test_data

# Paths
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model2_results.csv"

HF_REPO      = "FrankAlonsoskyMolina/clearsight-models"
HF_SUBFOLDER = "model2_deep_learning/saved_model"


def ensure_model_files():
    """Download all saved_model files from HuggingFace if any are missing."""
    if not any(SAVED_MODEL_DIR.glob("*.joblib")) or not (SAVED_MODEL_DIR / "model.keras").exists():
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
    """Load trained DNN model, scaler, preprocessing state, and feature names."""
    ensure_model_files()
    model = tf.keras.models.load_model(SAVED_MODEL_DIR / "model.keras")
    scaler = joblib.load(SAVED_MODEL_DIR / "scaler.joblib")
    preprocessing_state = joblib.load(SAVED_MODEL_DIR / "preprocessing_state.joblib")
    feature_names = joblib.load(SAVED_MODEL_DIR / "feature_names.joblib")
    return model, scaler, preprocessing_state, feature_names


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


def predict(model, scaler, X_test, feature_names):
    """Generate predictions and format output."""
    # Align columns to training order
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # Scale using training scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_proba = model.predict(X_test_scaled).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # Confidence = how sure the model is
    confidence = np.maximum(y_proba, 1 - y_proba)

    return y_pred, y_proba, confidence


def main():
    # 1. Load model and artifacts
    print("Loading model...")
    model, scaler, preprocessing_state, feature_names = load_model()

    # 2. Find and load test data
    test_file = find_test_file()
    print(f"Test file found: {test_file}")

    # 3. Preprocess test data using same pipeline as training
    #    prepare_test_data returns aligned IDs (safe after sort + filter)
    X_test, ids = prepare_test_data(str(test_file), preprocessing_state)

    # 4. Generate predictions
    y_pred, y_proba, confidence = predict(model, scaler, X_test, feature_names)

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