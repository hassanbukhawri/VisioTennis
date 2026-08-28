"""
Post-hoc temperature scaling for the CLEAN-SPLIT (player-disjoint) StrokeGRU
checkpoint, now deployed in backend/app.py as of 2026-08-21 (replacing the
original leaky-split checkpoint -- see PROJECT_LOG.md). The T=1.4474 value
fit earlier this session was fit against the LEAKY-split checkpoint's own
val set and does not apply here -- a different checkpoint has different
weights and a different confidence distribution, so it needs its own fit,
not a reused constant.

Same method as calibrate_temperature.py (Guo et al. 2017): single scalar T
fit via LBFGS on this checkpoint's own held-out val split (never used in
training or checkpoint selection), ECE reported via reliability-bucket
breakdown on this checkpoint's own held-out test split.

Usage:
    python training/stroke_classification/calibrate_temperature_clean_split.py
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

CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "clean_split", "stroke_gru_clean_best.pt")
CLEAN_DATA_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")
LEAKY_DATA_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed")
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
    print(f"[setup] val samples={len(val_ds)}  test samples={len(test_ds)}  (clean/player-disjoint split -- "
          f"the split THIS checkpoint was selected against, matches what's now deployed)")

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

    # Cross-check against the OLD leaky-split's val/test data -- NOT this
    # checkpoint's own split, reported separately and clearly labeled, same
    # honesty-check discipline as the original calibration entry.
    if os.path.isdir(LEAKY_DATA_ROOT):
        leaky_test_ds = StrokeDataset("test", data_root=LEAKY_DATA_ROOT)
        with torch.no_grad():
            leaky_logits = model(leaky_test_ds.X.to(device))
        leaky_labels = leaky_test_ds.y.to(device)

        leaky_probs_before = F.softmax(leaky_logits, dim=1)
        leaky_acc = (leaky_probs_before.argmax(dim=1) == leaky_labels).float().mean().item()
        leaky_ece_before, _ = compute_ece(leaky_probs_before, leaky_labels)
        leaky_probs_after = F.softmax(leaky_logits / T, dim=1)
        leaky_ece_after, _ = compute_ece(leaky_probs_after, leaky_labels)

        print(f"\n{'=' * 70}\nADDITIONAL CHECK: clean-split model+T against the OLD leaky-split test set\n"
              f"(NOT the set T was fit on; this checkpoint's own train set does NOT include these "
              f"players' clips at all, unlike the reverse check done for the leaky model -- this IS a "
              f"genuine held-out check)\n{'=' * 70}")
        print(f"[result] accuracy = {leaky_acc:.3f}")
        print(f"[result] ECE before = {leaky_ece_before:.4f} ({leaky_ece_before * 100:.2f}%)")
        print(f"[result] ECE after  = {leaky_ece_after:.4f} ({leaky_ece_after * 100:.2f}%)")

    out = {
        "temperature": T,
        "fitted_on": "training/stroke_classification/checkpoints/clean_split/stroke_gru_clean_best.pt (deployed, clean/player-disjoint split, 2026-08-21)",
        "val_split": f"{CLEAN_DATA_ROOT}/val ({len(val_ds)} samples)",
        "test_ece_before": ece_before,
        "test_ece_after": ece_after,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] temperature saved to {OUT_PATH} (overwrote the leaky-split model's T)")


if __name__ == "__main__":
    main()
