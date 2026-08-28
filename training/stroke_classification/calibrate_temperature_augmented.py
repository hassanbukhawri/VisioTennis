"""
Post-hoc temperature scaling for the THETIS-augmentation checkpoint, deployed
in backend/app.py as of 2026-08-22 (replacing the clean-split checkpoint --
see PROJECT_LOG.md). Neither the leaky-split T=1.4474 nor the clean-split
T=2.0046 apply here -- confirmed twice now that T does not transfer between
checkpoints, so this one gets its own fit, same as the other two.

val/test for calibration purposes are the UNCHANGED processed_clean val/test
(processed_clean_augmented only ever augmented the TRAIN partition -- see
build_augmented_train_set.py) -- synthetic data has no business in a
calibration fit or its evaluation either, same discipline as everywhere else
in this project.

Usage:
    python training/stroke_classification/calibrate_temperature_augmented.py
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from dataset import StrokeDataset, COLLAPSED_CLASSES
from model import StrokeGRU
from calibrate_temperature import compute_ece, print_reliability_table, fit_temperature

CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "augmented", "stroke_gru_augmented_best.pt")
CLEAN_DATA_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")
OUT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "temperature.json")


def main():
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"[setup] loaded deployed checkpoint: {CKPT_PATH}")
    print(f"[setup] checkpoint epoch={ckpt['epoch']} val_macro_f1={ckpt['val_macro_f1']:.3f}")

    model = StrokeGRU(input_dim=99, num_classes=len(COLLAPSED_CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = StrokeDataset("val", data_root=CLEAN_DATA_ROOT)
    test_ds = StrokeDataset("test", data_root=CLEAN_DATA_ROOT)
    print(f"[setup] val samples={len(val_ds)}  test samples={len(test_ds)}  "
          f"(unaugmented clean-split val/test -- this checkpoint's real held-out sets)")

    with torch.no_grad():
        val_logits = model(val_ds.X.to(device))
        test_logits = model(test_ds.X.to(device))
    val_labels = val_ds.y.to(device)
    test_labels = test_ds.y.to(device)

    probs_before = F.softmax(test_logits, dim=1)
    acc = (probs_before.argmax(dim=1) == test_labels).float().mean().item()
    ece_before, rows_before = compute_ece(probs_before, test_labels)
    print(f"\n{'=' * 70}\nBEFORE calibration (raw softmax, T=1) -- clean-split test set\n{'=' * 70}")
    print(f"[result] accuracy = {acc:.3f}")
    print(f"[result] ECE = {ece_before:.4f} ({ece_before * 100:.2f}%)")
    print_reliability_table("clean-split test, T=1", rows_before, len(test_ds))

    T = fit_temperature(val_logits, val_labels)

    probs_after = F.softmax(test_logits / T, dim=1)
    ece_after, rows_after = compute_ece(probs_after, test_labels)
    print(f"\n{'=' * 70}\nAFTER calibration (T={T:.4f}) -- clean-split test set\n{'=' * 70}")
    print(f"[result] accuracy = {(probs_after.argmax(dim=1) == test_labels).float().mean().item():.3f} "
          f"(should match BEFORE exactly)")
    print(f"[result] ECE = {ece_after:.4f} ({ece_after * 100:.2f}%)")
    print_reliability_table(f"clean-split test, T={T:.2f}", rows_after, len(test_ds))

    print(f"\n[summary] ECE: {ece_before:.4f} -> {ece_after:.4f} "
          f"({'IMPROVED' if ece_after < ece_before else 'WORSE'})")

    out = {
        "temperature": T,
        "fitted_on": "training/stroke_classification/checkpoints/augmented/stroke_gru_augmented_best.pt (deployed, 2026-08-22)",
        "val_split": f"{CLEAN_DATA_ROOT}/val ({len(val_ds)} samples, unaugmented)",
        "test_ece_before": ece_before,
        "test_ece_after": ece_after,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] temperature saved to {OUT_PATH} (overwrote the clean-split model's T)")


if __name__ == "__main__":
    main()
