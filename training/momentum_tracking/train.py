"""
Model 3 (momentum tracking) -- training script. First real training loop for
TennisMomentumLSTM (model.py), which until now was architecture-only (per
its own docstring: "No training loop, no dataset class... do not wire this
into a training script until real data is acquired").

Real data is now in hand (dataset.py, ~4,994 real-ATP-linked matches,
823,874 points, EWMA-labeled and sanity-checked). This script is the actual
wiring: match-level train/val/test split, variable-length padded/masked
sequences, masked MSE loss (padded timesteps must not contribute), smoke
test before any full run.

Usage:
    python training/momentum_tracking/train.py --smoke-test
    python training/momentum_tracking/train.py --epochs 30 --batch-size 32
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from dataset import (
    load_and_engineer, match_level_split, MomentumDataset, collate_pad, N_FEATURES, DEFAULT_HORIZON,
)
from model import TennisMomentumLSTM

CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help="points ahead the target is shifted (leak-free reframe, 2026-08-26)")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-matches", type=int, default=64)
    p.add_argument("--min-points", type=int, default=20)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--ckpt-name", default="momentum_lstm_best.pt")
    p.add_argument("--last-ckpt-name", default="momentum_lstm_last.pt")
    return p.parse_args()


class MaskedMSELoss(nn.Module):
    """Padded timesteps carry no real point (all-zero feature rows, target
    0.0 by construction in collate_pad) and must not contribute to the
    loss -- otherwise the model would be rewarded/penalized for its output
    on positions that don't correspond to a real point at all."""

    def forward(self, pred, target, mask):
        pred = pred.squeeze(-1)  # (B, T)
        se = (pred - target) ** 2
        se = se * mask.float()
        return se.sum() / mask.float().sum().clamp(min=1)


def masked_mae(pred, target, mask):
    pred = pred.squeeze(-1)
    ae = (pred - target).abs() * mask.float()
    return (ae.sum() / mask.float().sum().clamp(min=1)).item()


def run_epoch(model, loader, device, optimizer, loss_fn, train):
    model.train(mode=train)
    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    for x, y, mask, lengths in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with torch.set_grad_enabled(train):
            pred = model(x, lengths)
            loss = loss_fn(pred, y, mask)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_mae += masked_mae(pred, y, mask)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mae / max(n_batches, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("[setup] loading + engineering features from real point data...", flush=True)
    t0 = time.time()
    featured = load_and_engineer(min_points=args.min_points)
    print(f"[setup] {featured['match_id'].nunique()} matches, {len(featured)} points "
          f"({time.time() - t0:.1f}s)", flush=True)

    all_match_ids = featured["match_id"].unique().tolist()
    train_ids, val_ids, test_ids = match_level_split(all_match_ids)
    print(f"[setup] match-level split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
          f"(verified zero match_id overlap in match_level_split's own assertion)", flush=True)

    if args.smoke_test:
        import random
        random.seed(0)
        train_ids = set(random.sample(sorted(train_ids), min(args.smoke_matches, len(train_ids))))
        val_ids = set(random.sample(sorted(val_ids), min(max(8, args.smoke_matches // 4), len(val_ids))))
        print(f"[smoke] subsampled to train={len(train_ids)} val={len(val_ids)} matches", flush=True)

    train_ds = MomentumDataset(featured, train_ids, horizon=args.horizon)
    val_ds = MomentumDataset(featured, val_ids, horizon=args.horizon)
    print(f"[setup] horizon={args.horizon} (each position's target is the EWMA momentum value "
          f"{args.horizon} point(s) after the state its input describes -- horizon=1 predicts the "
          f"very next point's own resulting momentum without seeing that point's outcome; leak-free "
          f"reframe, 2026-08-26, see dataset.py module docstring)", flush=True)
    print(f"[setup] train_ds={len(train_ds)} matches, val_ds={len(val_ds)} matches", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_pad)

    model = TennisMomentumLSTM(input_dim=N_FEATURES, hidden_dim=args.hidden_dim,
                                num_layers=args.num_layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] TennisMomentumLSTM params: {n_params / 1e6:.3f}M (input_dim={N_FEATURES}, "
          f"leak-free feature set -- server + score-state entering the point only, all "
          f"point-outcome-derived features excluded; see dataset.py's module docstring)", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MaskedMSELoss()

    epochs = 5 if args.smoke_test else args.epochs
    best_val_mae = float("inf")
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    last_ckpt_path = os.path.join(args.ckpt_dir, args.last_ckpt_name)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_mae = run_epoch(model, train_loader, device, optimizer, loss_fn, train=True)
        train_time = time.time() - t0

        t1 = time.time()
        val_loss, val_mae = run_epoch(model, val_loader, device, optimizer, loss_fn, train=False)
        val_time = time.time() - t1

        mem_str = ""
        if device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated() / 1e9
            mem_str = f" | peak_alloc={alloc:.2f}GB"

        print(
            f"[epoch {epoch}/{epochs}] train_mse={train_loss:.4f} train_mae={train_mae:.4f} | "
            f"val_mse={val_loss:.4f} val_mae={val_mae:.4f} | "
            f"train_time={train_time:.1f}s val_time={val_time:.1f}s{mem_str}",
            flush=True,
        )

        if not args.smoke_test and not args.no_checkpoint:
            ckpt_dict = {
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "args": vars(args), "epoch": epoch, "val_mse": val_loss, "val_mae": val_mae,
                "n_features": N_FEATURES, "horizon": args.horizon,
            }
            torch.save(ckpt_dict, last_ckpt_path)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(ckpt_dict, ckpt_path)
                print(f"[checkpoint] new best val_mae={val_mae:.4f} -> saved {ckpt_path}", flush=True)

    if args.smoke_test:
        print("[smoke] SMOKE TEST COMPLETE.", flush=True)
        if device.type == "cuda":
            print(f"[smoke] peak GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    else:
        print(f"[done] best val_mae={best_val_mae:.4f}, checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
