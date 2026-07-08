import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
DATASET_PATH = "dataset"


def build_val_dataset(dataset_path=DATASET_PATH):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    return val_ds


def evaluate_model(model_path: str, out_path: str = "saved_model/metrics.json"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)
    val_ds = build_val_dataset()

    y_true = []
    y_pred = []

    class_names = val_ds.class_names

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Compute macro averages if not present
    overall = {
        "accuracy": acc,
        "precision_macro": float(report.get("macro avg", {}).get("precision", 0.0)),
        "recall_macro": float(report.get("macro avg", {}).get("recall", 0.0)),
        "f1_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
    }

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            "precision": float(report.get(cls, {}).get("precision", 0.0)),
            "recall": float(report.get(cls, {}).get("recall", 0.0)),
            "f1-score": float(report.get(cls, {}).get("f1-score", 0.0)),
            "support": int(report.get(cls, {}).get("support", 0)),
        }

    metrics = {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": class_names,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {out_path}")
    return metrics


if __name__ == "__main__":
    model_path = os.path.join("saved_model", "garbage_classifier.keras")
    out_path = os.path.join("saved_model", "metrics.json")
    evaluate_model(model_path, out_path)
