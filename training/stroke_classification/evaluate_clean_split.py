"""
Test-set evaluation for the StrokeGRU clean-split retrain -- same
methodology as the original (leaky) 84.5%/0.834 numbers documented in
README.md (held-out test split, never seen in training or checkpoint
selection, checkpoint chosen on best val macro-F1), so the two numbers are
directly comparable. See PROJECT_LOG.md, 2026-08-20, for why the retrain
happened: the original split had 100% player overlap across train/val/test.

Usage:
    python training/stroke_classification/evaluate_clean_split.py
"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from dataset import StrokeDataset, COLLAPSED_CLASSES
from model import StrokeGRU

DATA_ROOT = os.path.join("data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "clean_split", "stroke_gru_clean_best.pt")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"[setup] checkpoint epoch={ckpt['epoch']} val_macro_f1={ckpt['val_macro_f1']:.3f}")

    model = StrokeGRU(input_dim=99, num_classes=len(COLLAPSED_CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = StrokeDataset("test", data_root=DATA_ROOT)
    print(f"[setup] test samples={len(test_ds)} class_counts={test_ds.class_counts().tolist()}")

    X = test_ds.X.to(device)
    y = test_ds.y.numpy()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = float((preds == y).mean())
    macro_f1 = f1_score(y, preds, average="macro", labels=list(range(len(COLLAPSED_CLASSES))))

    print(f"\n[result] test accuracy = {acc:.3f} ({acc*100:.1f}%)")
    print(f"[result] test macro-F1 = {macro_f1:.3f}")
    print("\n" + classification_report(y, preds, target_names=COLLAPSED_CLASSES, digits=3))

    cm = confusion_matrix(y, preds, labels=list(range(len(COLLAPSED_CLASSES))))
    print("Confusion matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(10) + "".join(c[:8].rjust(10) for c in COLLAPSED_CLASSES)
    print(header)
    for i, c in enumerate(COLLAPSED_CLASSES):
        print(c.ljust(10) + "".join(str(v).rjust(10) for v in cm[i]))

    out_path = os.path.join(SCRIPT_DIR, "checkpoints", "clean_split", "test_confusion_matrix.npy")
    np.save(out_path, cm)
    print(f"\n[done] confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
