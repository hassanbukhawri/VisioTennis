"""
Model 3 (momentum tracking) -- held-out test evaluation + qualitative
check against the same 3 hand-picked real matches sanity_check_ewma.py
already validated the EWMA LABEL against (Gaudio comeback, Federer/
Wawrinka, Djokovic/Cilic negative control). Uses the BEST checkpoint by
val_mae (not last epoch), per the same overfitting-awareness standard as
every other model in this project.

Usage:
    python training/momentum_tracking/evaluate.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from dataset import (
    load_and_engineer, match_level_split, MomentumDataset, collate_pad,
    FEATURE_NAMES, N_FEATURES, DEFAULT_HORIZON,
)
from model import TennisMomentumLSTM

CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "momentum_lstm_best.pt")
FIGS_DIR = os.path.join(SCRIPT_DIR, "figures")

EXAMPLE_MATCHES = {
    "20040606-M-Roland_Garros-F-Gaston_Gaudio-Guillermo_Coria": (
        "2004 Roland Garros Final: Gaudio d. Coria, 0-6 3-6 6-4 6-1 8-6\n"
        "(real comeback -- EWMA label: sustained negative through ~pt140, crosses positive, Gaudio wins)"
    ),
    "20150516-M-Rome_Masters-SF-Stan_Wawrinka-Roger_Federer": (
        "2015 Rome Masters SF: Federer d. Wawrinka, 6-4 6-2\n"
        "(EWMA label: early Wawrinka lead, Federer's momentum turns around)"
    ),
    "20150911-M-US_Open-SF-Novak_Djokovic-Marin_Cilic": (
        "2015 US Open SF: Djokovic d. Cilic, 6-0 6-1 6-2\n(negative control: dominant win, EWMA label stays positive throughout, no reversal)"
    ),
}


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = TennisMomentumLSTM(input_dim=ckpt.get("n_features", N_FEATURES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    horizon = ckpt.get("horizon", DEFAULT_HORIZON)
    print(f"[setup] loaded checkpoint: epoch={ckpt['epoch']} val_mae={ckpt['val_mae']:.4f} "
          f"val_mse={ckpt['val_mse']:.4f} horizon={horizon}")
    return model, horizon


def evaluate_test_set(model, featured, test_ids, device, horizon):
    test_ds = MomentumDataset(featured, test_ids, horizon=horizon)
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

    total_se, total_ae, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, mask, lengths in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x, lengths).squeeze(-1)
            se = ((pred - y) ** 2) * mask.float()
            ae = (pred - y).abs() * mask.float()
            total_se += se.sum().item()
            total_ae += ae.sum().item()
            total_n += mask.sum().item()

    mse = total_se / total_n
    mae = total_ae / total_n
    print(f"\n[test set] n_matches={len(test_ds)} n_points={int(total_n)}")
    print(f"[test set] MSE={mse:.5f}  MAE={mae:.5f}  RMSE={mse**0.5:.5f}")
    return mse, mae


def qualitative_check(model, featured, device, horizon):
    """Plots pred[i] against the SAME target it was trained to predict --
    true_ewma at point (Pt[i] + horizon - 1), not true_ewma[i] itself (that
    would compare a forecast to the wrong point, understating error on a
    trending match and overstating it on a flat one). x is truncated by the
    same (horizon-1) trailing points MomentumDataset drops, for the same
    reason (no valid future target past the match's last point)."""
    FIGS_DIR_ = FIGS_DIR
    os.makedirs(FIGS_DIR_, exist_ok=True)
    fig, axes = plt.subplots(len(EXAMPLE_MATCHES), 1, figsize=(11, 3.6 * len(EXAMPLE_MATCHES)), sharex=False)

    for ax, (mid, title) in zip(axes, EXAMPLE_MATCHES.items()):
        sub = featured[featured["match_id"] == mid].sort_values("Pt")
        if sub.empty:
            ax.set_title(f"{title}\n[NOT FOUND -- dropped by min_points filter or not already_ingested]")
            continue
        t = len(sub)
        usable_t = t - (horizon - 1)
        if usable_t < 1:
            ax.set_title(f"{title}\n[too short for horizon={horizon}]")
            continue

        x_full = sub[FEATURE_NAMES].to_numpy(dtype=np.float32)
        x = torch.from_numpy(x_full[:usable_t]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, lengths=None).squeeze(0).squeeze(-1).cpu().numpy()

        true_ewma_full = sub["momentum_ewma_p1"].to_numpy()
        pts_full = sub["Pt"].to_numpy()
        # target/x-axis for pred[i] is the point horizon-1 steps AHEAD of x's own point i
        true_ewma_target = true_ewma_full[horizon - 1: horizon - 1 + usable_t]
        pts_target = pts_full[horizon - 1: horizon - 1 + usable_t]

        ax.plot(pts_full, true_ewma_full, color="gray", linewidth=1.0, linestyle="--",
                label="EWMA ground truth (label)")
        ax.plot(pts_target, pred, color="#2b6cb0", linewidth=1.3,
                label=f"model output (forecast, horizon={horizon})")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("momentum (P1 perspective)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)

        mse_this = float(np.mean((pred - true_ewma_target) ** 2))
        print(f"[{mid}] n_points={len(sub)} model-vs-EWMA(horizon={horizon}) MSE on this match={mse_this:.5f}")

    axes[-1].set_xlabel("point number in match (Pt)")
    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR_, "trained_model_vs_ewma_examples.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, horizon = load_model(device)

    featured = load_and_engineer()
    train_ids, val_ids, test_ids = match_level_split(featured["match_id"].unique().tolist())

    evaluate_test_set(model, featured, test_ids, device, horizon)
    qualitative_check(model, featured, device, horizon)


if __name__ == "__main__":
    main()
