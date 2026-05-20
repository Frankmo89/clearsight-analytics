import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
MODEL_PATH      = Path(__file__).resolve().parent / "saved_model"
TEST_DATA_DIR   = PROJECT_ROOT / "test_data"
OUTPUT_FILE     = TEST_DATA_DIR / "model3_results.csv"

HF_REPO      = "FrankAlonsoskyMolina/clearsight-analytics"
HF_SUBFOLDER = "model3_cnn/saved_model"


def ensure_model_files():
    """Download all saved_model files from HuggingFace if any are missing."""
    if not (MODEL_PATH / "best_model.keras").exists():
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
    ensure_model_files()
    model = tf.keras.models.load_model(
        MODEL_PATH / "best_model.keras",
        compile=False
    )
    print("Model loaded.")
    return model


def load_images(image_paths):
    images    = []
    valid_ids = []

    for img_id, path in image_paths.items():
        try:
            img = tf.keras.utils.load_img(path, target_size=(224, 224))
        except Exception as e:
            print(f"WARNING: Could not load {path}: {e}")
            continue

        img = tf.keras.utils.img_to_array(img)
        img = preprocess_input(img)

        images.append(img)
        valid_ids.append(img_id)

    return np.array(images), valid_ids


def predict(model, images, image_ids):
    preds = model(images).numpy()

    predicted_classes  = (preds > 0.5).astype(int).flatten()
    confidence_scores  = np.where(preds.flatten() > 0.5, preds.flatten(), 1 - preds.flatten())

    return pd.DataFrame({
        "image_id":        image_ids,
        "predicted_class": predicted_classes,
        "confidence":      confidence_scores,
    })


def main():
    model = load_model()

    test_image_dir    = TEST_DATA_DIR / "images"
    VALID_EXTENSIONS  = {".png", ".jpg", ".jpeg"}

    image_paths = {
        img_path.stem: img_path
        for img_path in sorted(test_image_dir.glob("*.*"))
        if img_path.suffix.lower() in VALID_EXTENSIONS
    }

    images, image_ids = load_images(image_paths)
    results = predict(model, images, image_ids)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()