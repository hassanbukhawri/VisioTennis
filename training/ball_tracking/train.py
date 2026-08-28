"""
Training script for VisioTennis Model 1A: TrackNet-style ball heatmap regression.

Usage:
    # smoke test (small subset, 1-2 epochs, sanity-check shapes/loss/memory)
    python training/ball_tracking/train.py --smoke-test

    # full run (launch as background process, see repo-level instructions)
    python training/ball_tracking/train.py --epochs 60 --batch-size 24
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BallTrackingDataset, discover_clips, split_by_game, BALL_ROOT
from model import TrackNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=BALL_ROOT)
    p.add_argument("--seq-len", type=int, default=3)
    p.add_argument("--out-w", type=int, default=512)
    p.add_argument("--out-h", type=int, default=288)
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian heatmap sigma, in OUTPUT-res pixels")
    p.add_argument("--stride", type=int, default=1, help="sliding-window stride within a clip")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos-weight", type=float, default=100.0, help="weighted-MSE emphasis on ball pixels")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dist-thresh", type=float, default=4.0, help="output-res px threshold for PCK metric")
    p.add_argument("--conf-thresh", type=float, default=0.5, help="heatmap-max threshold for 'ball present'")

    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-samples", type=int, default=80)
    p.add_argument("--max-train-samples", type=int, default=None,
                    help="cap train set size without forcing smoke-test num_workers=0 "
                         "(useful for a realistic throughput calibration run)")
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--no-checkpoint", action="store_true",
                    help="skip saving checkpoints (e.g. for a calibration run)")
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--ckpt-name", default="tracknet_best.pt")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def weighted_mse_loss(pred, target, pos_weight):
    """Plain MSE collapses to predicting all-zero heatmaps because >99% of
    pixels are background. Upweight pixels near the Gaussian peak so the
    (rare) ball pixels aren't drowned out."""
    weight = 1.0 + pos_weight * target
    return (weight * (pred - target) ** 2).mean()


@torch.no_grad()
def batch_peak_coords(heatmaps):
    """heatmaps: (B, T, H, W) -> peak (x, y) per (B, T), plus the peak value."""
    B, T, H, W = heatmaps.shape
    flat = heatmaps.view(B, T, -1)
    peak_val, peak_idx = flat.max(dim=-1)
    py = (peak_idx // W).float()
    px = (peak_idx % W).float()
    return px, py, peak_val


@torch.no_grad()
def compute_metrics(pred, target, visibility, dist_thresh, conf_thresh):
    """
    pred, target: (B, T, H, W)
    visibility:   (B, T) raw code 0..3 (0 = not visible)
    Returns dict with pck (distance-based accuracy over visible frames) and
    no_ball_acc (correct "no ball" predictions over invisible frames).
    """
    px, py, pval = batch_peak_coords(pred)
    gx, gy, _ = batch_peak_coords(target)

    vis_mask = (visibility != 0)
    dist = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)

    n_vis = vis_mask.sum().item()
    if n_vis > 0:
        hits = ((dist <= dist_thresh) & vis_mask).sum().item()
        pck = hits / n_vis
    else:
        pck = float("nan")

    invis_mask = ~vis_mask
    n_invis = invis_mask.sum().item()
    if n_invis > 0:
        no_ball_hits = ((pval < conf_thresh) & invis_mask).sum().item()
        no_ball_acc = no_ball_hits / n_invis
    else:
        no_ball_acc = float("nan")

    return {"pck": pck, "n_vis": n_vis, "no_ball_acc": no_ball_acc, "n_invis": n_invis}


def run_epoch(model, loader, device, optimizer, scaler, pos_weight, dist_thresh, conf_thresh, train):
    model.train(mode=train)
    total_loss, total_vis_hits, total_vis, total_invis_hits, total_invis = 0.0, 0, 0, 0, 0
    n_batches = 0

    for frames, heatmaps, visibility in loader:
        frames = frames.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        visibility = visibility.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(frames)
                loss = weighted_mse_loss(pred, heatmaps, pos_weight)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        m = compute_metrics(pred.float(), heatmaps.float(), visibility, dist_thresh, conf_thresh)
        total_loss += loss.item()
        n_batches += 1
        if m["n_vis"] > 0:
            total_vis_hits += m["pck"] * m["n_vis"]
            total_vis += m["n_vis"]
        if m["n_invis"] > 0:
            total_invis_hits += m["no_ball_acc"] * m["n_invis"]
            total_invis += m["n_invis"]

    avg_loss = total_loss / max(n_batches, 1)
    pck = total_vis_hits / total_vis if total_vis > 0 else float("nan")
    no_ball_acc = total_invis_hits / total_invis if total_invis > 0 else float("nan")
    return avg_loss, pck, no_ball_acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"[setup] discovering clips under {args.data_root!r} ...", flush=True)
    clips = discover_clips(args.data_root)
    train_clips, val_clips, val_games = split_by_game(
        clips, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"[setup] {len(clips)} clips total -> {len(train_clips)} train clips / "
          f"{len(val_clips)} val clips. val_games={val_games}", flush=True)

    if args.smoke_test:
        max_train = args.max_train_samples or args.smoke_samples
        max_val = args.max_val_samples or max(8, args.smoke_samples // 4)
    else:
        max_train = args.max_train_samples
        max_val = args.max_val_samples

    train_ds = BallTrackingDataset(
        train_clips, seq_len=args.seq_len, out_size=(args.out_w, args.out_h),
        sigma=args.sigma, stride=args.stride, max_samples=max_train,
    )
    val_ds = BallTrackingDataset(
        val_clips, seq_len=args.seq_len, out_size=(args.out_w, args.out_h),
        sigma=args.sigma, stride=args.stride, max_samples=max_val,
    )
    print(f"[setup] train samples={len(train_ds)}, val samples={len(val_ds)}", flush=True)

    num_workers = 0 if args.smoke_test else args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = TrackNet(seq_len=args.seq_len, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if args.resume:
        print(f"[setup] resuming from {args.resume}", flush=True)
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            print("[setup] restored optimizer state", flush=True)
        else:
            print("[setup] checkpoint has no optimizer_state (older format) -- optimizer starts fresh", flush=True)
        if "scaler_state" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler_state"])

    epochs = 2 if args.smoke_test else args.epochs
    best_pck = -1.0
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)

    # One quick forward pass up front to confirm shapes before any training.
    sample_frames, sample_hm, sample_vis = next(iter(train_loader))
    print(f"[smoke] frame batch shape={tuple(sample_frames.shape)}, "
          f"heatmap batch shape={tuple(sample_hm.shape)}, "
          f"visibility batch shape={tuple(sample_vis.shape)}", flush=True)
    assert sample_frames.shape[1] == 3 * args.seq_len
    assert sample_hm.shape[1] == args.seq_len
    assert sample_frames.shape[-2:] == (args.out_h, args.out_w)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_pck, train_no_ball = run_epoch(
            model, train_loader, device, optimizer, scaler,
            args.pos_weight, args.dist_thresh, args.conf_thresh, train=True,
        )
        train_time = time.time() - t0

        t1 = time.time()
        val_loss, val_pck, val_no_ball = run_epoch(
            model, val_loader, device, optimizer, scaler,
            args.pos_weight, args.dist_thresh, args.conf_thresh, train=False,
        )
        val_time = time.time() - t1

        mem_str = ""
        if device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" | peak_alloc={alloc:.2f}GB peak_reserved={reserved:.2f}GB"

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.5f} train_pck@{args.dist_thresh:.0f}px={train_pck:.3f} "
            f"train_no_ball_acc={train_no_ball:.3f} | "
            f"val_loss={val_loss:.5f} val_pck@{args.dist_thresh:.0f}px={val_pck:.3f} "
            f"val_no_ball_acc={val_no_ball:.3f} | "
            f"train_time={train_time:.1f}s val_time={val_time:.1f}s{mem_str}",
            flush=True,
        )

        if not args.smoke_test and not args.no_checkpoint and val_pck > best_pck:
            best_pck = val_pck
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_pck": val_pck,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"[checkpoint] new best val_pck={val_pck:.3f} -> saved {ckpt_path}", flush=True)

    if args.smoke_test:
        print("[smoke] SMOKE TEST COMPLETE -- shapes OK, loop ran without error.", flush=True)
        if device.type == "cuda":
            print(f"[smoke] peak GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    else:
        print(f"[done] best val_pck={best_pck:.3f}, checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
