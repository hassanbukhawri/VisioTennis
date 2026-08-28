"""
Post-hoc temperature scaling for the StrokeGRU model actually deployed in
backend/app.py today: `checkpoints/stroke_gru_best.pt` / `stroke_gru.onnx`,
the ORIGINAL leaky-split checkpoint (README.md: "deliverable checkpoints").
NOT the clean-split retrain (checkpoints/clean_split/) -- that checkpoint
has never been exported to ONNX and is not wired into the live pipeline, so
calibrating it would calibrate a model nobody is actually running. See
PROJECT_LOG.md, 2026-08-21 entry, for the confirmation that surfaced this
distinction and the user's explicit decision to calibrate the live model.

Standard approach (Guo et al. 2017, "On Calibration of Modern Neural
Networks"): a single scalar T > 1 that softens logits (softmax(z/T))
without changing argmax/accuracy, fit by minimizing NLL on a held-out
validation set via LBFGS. Reported against: reliability-diagram-style ECE
(equal-width confidence bins, weighted |accuracy - confidence| per bin) on
the model's own held-out TEST split (never used for fitting T).

Also runs the same before/after comparison against the clean-split test set
(player-disjoint) as an additional, more-honest-about-overlap check -- NOT
the set T is fit on, reported separately and labeled as such.

Usage:
    python training/stroke_classification/calibrate_temperature.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from dataset import StrokeDataset, COLLAPSED_CLASSES, DATA_ROOT as LEAKY_DATA_ROOT
from model import StrokeGRU

CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "stroke_gru_best.pt")
CLEAN_DATA_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")
OUT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "temperature.json")

N_BINS = 10


def compute_ece(probs, labels, n_bins=N_BINS):
    """Equal-width confidence-bin ECE. probs: (N, C) softmax output, labels: (N,)."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        frac_in_bin = in_bin.float().mean().item()
        if in_bin.sum().item() == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        acc_in_bin = accuracies[in_bin].float().mean().item()
        conf_in_bin = confidences[in_bin].mean().item()
        ece += torch.abs(torch.tensor(conf_in_bin - acc_in_bin)) * frac_in_bin
        rows.append((lo, hi, int(in_bin.sum().item()), acc_in_bin, conf_in_bin))
    return ece.item(), rows


def print_reliability_table(label, rows, n_total):
    print(f"\n[reliability] {label} (n={n_total})")
    print(f"  {'bucket':>12}  {'n':>5}  {'accuracy':>9}  {'avg conf':>9}  {'gap':>7}")
    for lo, hi, n, acc, conf in rows:
        if n == 0:
            print(f"  {lo:.1f}-{hi:.1f}      {n:>5}  {'--':>9}  {'--':>9}  {'--':>7}")
        else:
            gap = conf - acc
            print(f"  {lo:.1f}-{hi:.1f}      {n:>5}  {acc:>8.1%}  {conf:>8.1%}  {gap:>+6.1%}")


def fit_temperature(val_logits, val_labels):
    """LBFGS fit of a single scalar T minimizing NLL on val_logits/T vs val_labels."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)
    nll_before = F.cross_entropy(val_logits, val_labels).item()

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / temperature, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = temperature.item()
    nll_after = F.cross_entropy(val_logits / temperature, val_labels).item()
    print(f"\n[fit] val NLL before scaling: {nll_before:.4f}")
    print(f"[fit] fitted temperature T = {T:.4f}")
    print(f"[fit] val NLL after scaling:  {nll_after:.4f}")
    return T


def main():
    device = torch.device("cpu")  # calibration is a one-off analysis step, no need for GPU
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"[setup] loaded deployed checkpoint: {CKPT_PATH}")
    print(f"[setup] checkpoint epoch={ckpt['epoch']} val_macro_f1={ckpt['val_macro_f1']:.3f}")

    model = StrokeGRU(input_dim=99, num_classes=len(COLLAPSED_CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = StrokeDataset("val", data_root=LEAKY_DATA_ROOT)
    test_ds = StrokeDataset("test", data_root=LEAKY_DATA_ROOT)
    print(f"[setup] val samples={len(val_ds)}  test samples={len(test_ds)}  (original/leaky split -- "
          f"the split this exact checkpoint was selected against, matches what's deployed)")

    with torch.no_grad():
        val_logits = model(val_ds.X.to(device))
        test_logits = model(test_ds.X.to(device))
    val_labels = val_ds.y.to(device)
    test_labels = test_ds.y.to(device)

    # --- BEFORE calibration (T=1, i.e. plain softmax -- what's live today) ---
    probs_before = F.softmax(test_logits, dim=1)
    acc = (probs_before.argmax(dim=1) == test_labels).float().mean().item()
    ece_before, rows_before = compute_ece(probs_before, test_labels)
    print(f"\n{'=' * 70}\nBEFORE calibration (raw softmax, T=1) -- leaky-split test set\n{'=' * 70}")
    print(f"[result] accuracy = {acc:.3f} (unchanged by temperature scaling -- argmax-preserving)")
    print(f"[result] ECE = {ece_before:.4f} ({ece_before * 100:.2f}%)")
    print_reliability_table("leaky-split test, T=1", rows_before, len(test_ds))

    # --- fit T on val ---
    T = fit_temperature(val_logits, val_labels)

    # --- AFTER calibration ---
    probs_after = F.softmax(test_logits / T, dim=1)
    ece_after, rows_after = compute_ece(probs_after, test_labels)
    print(f"\n{'=' * 70}\nAFTER calibration (T={T:.4f}) -- leaky-split test set\n{'=' * 70}")
    print(f"[result] accuracy = {(probs_after.argmax(dim=1) == test_labels).float().mean().item():.3f} "
          f"(should match BEFORE exactly)")
    print(f"[result] ECE = {ece_after:.4f} ({ece_after * 100:.2f}%)")
    print_reliability_table(f"leaky-split test, T={T:.2f}", rows_after, len(test_ds))

    print(f"\n[summary] ECE: {ece_before:.4f} -> {ece_after:.4f} "
          f"({'IMPROVED' if ece_after < ece_before else 'WORSE'}, "
          f"{'' if ece_after < ece_before else 'not '}reduced by {(ece_before - ece_after):.4f})")

    # --- additional check: clean-split test set (player-disjoint), same model+T ---
    # NOT the set T was fit on -- this is honesty-about-overlap check only, reported
    # separately and clearly labeled, per the original ask.
    if os.path.isdir(CLEAN_DATA_ROOT):
        clean_test_ds = StrokeDataset("test", data_root=CLEAN_DATA_ROOT)
        with torch.no_grad():
            clean_logits = model(clean_test_ds.X.to(device))
        clean_labels = clean_test_ds.y.to(device)

        clean_probs_before = F.softmax(clean_logits, dim=1)
        clean_acc = (clean_probs_before.argmax(dim=1) == clean_labels).float().mean().item()
        clean_ece_before, clean_rows_before = compute_ece(clean_probs_before, clean_labels)

        clean_probs_after = F.softmax(clean_logits / T, dim=1)
        clean_ece_after, clean_rows_after = compute_ece(clean_probs_after, clean_labels)

        print(f"\n{'=' * 70}\nADDITIONAL CHECK: same model+T against the CLEAN-SPLIT test set\n"
              f"(player-disjoint from train -- NOT the set T was fit on, reported for honesty)\n{'=' * 70}")
        print(f"[note] this deployed (leaky-split-trained) model was never re-evaluated on "
              f"player-disjoint data before -- accuracy here is expected to be lower than the "
              f"leaky-split test set above, same reason the clean-split retrain exists.")
        print(f"[result] accuracy = {clean_acc:.3f}")
        print(f"[result] ECE before = {clean_ece_before:.4f} ({clean_ece_before * 100:.2f}%)")
        print(f"[result] ECE after  = {clean_ece_after:.4f} ({clean_ece_after * 100:.2f}%)")
        print_reliability_table(f"clean-split test, T=1", clean_rows_before, len(clean_test_ds))
        print_reliability_table(f"clean-split test, T={T:.2f}", clean_rows_after, len(clean_test_ds))
    else:
        print(f"\n[skip] clean-split data not found at {CLEAN_DATA_ROOT}, skipping additional check")

    # --- save T for backend/app.py to load ---
    out = {
        "temperature": T,
        "fitted_on": "training/stroke_classification/checkpoints/stroke_gru_best.pt (deployed, leaky-split)",
        "val_split": f"{LEAKY_DATA_ROOT}/val ({len(val_ds)} samples)",
        "test_ece_before": ece_before,
        "test_ece_after": ece_after,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] temperature saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
