"""
Training script for VisioTennis Model 1B: court keypoint direct regression.

Usage:
    python training/court_keypoints/train.py --smoke-test
    python training/court_keypoints/train.py --epochs 40 --batch-size 64
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CourtKeypointsDataset, KP_ROOT, N_KEYPOINTS
from model import CourtKeypointNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-json", default=os.path.join(KP_ROOT, "data_train.json"))
    p.add_argument("--val-json", default=os.path.join(KP_ROOT, "data_val.json"))
    p.add_argument("--images-dir", default=os.path.join(KP_ROOT, "images"))
    p.add_argument("--out-size", type=int, default=224)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pck-thresh-px", type=float, default=10.0, help="native-res px threshold for PCK")

    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-samples", type=int, default=80)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--ckpt-name", default="court_kp_best.pt")
    p.add_argument("--last-ckpt-name", default="court_kp_last.pt",
                   help="saved every epoch (regardless of improvement) so a crash never loses more than one epoch")
    p.add_argument("--resume", default=None)
    return p.parse_args()


@torch.no_grad()
def compute_metrics(pred, target, pck_thresh_px):
    """pred, target: (B, 2*N_KEYPOINTS) normalized to [0, 1].
    Returns mean native-res pixel distance and PCK (fraction of keypoints within
    pck_thresh_px), both averaged over all keypoints in the batch."""
    from dataset import NATIVE_W, NATIVE_H
    B = pred.shape[0]
    pred = pred.view(B, N_KEYPOINTS, 2)
    target = target.view(B, N_KEYPOINTS, 2)
    diff = pred - target
    dx = diff[..., 0] * NATIVE_W
    dy = diff[..., 1] * NATIVE_H
    dist = torch.sqrt(dx ** 2 + dy ** 2)  # (B, N_KEYPOINTS) in native px
    mean_dist = dist.mean().item()
    pck = (dist <= pck_thresh_px).float().mean().item()
    return mean_dist, pck


def run_epoch(model, loader, device, optimizer, scaler, loss_fn, pck_thresh_px, train):
    model.train(mode=train)
    total_loss, total_dist, total_pck, n_batches = 0.0, 0.0, 0.0, 0

    for images, kps in loader:
        images = images.to(device, non_blocking=True)
        kps = kps.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(images)
                loss = loss_fn(pred, kps)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        mean_dist, pck = compute_metrics(pred.float(), kps.float(), pck_thresh_px)
        total_loss += loss.item()
        total_dist += mean_dist
        total_pck += pck
        n_batches += 1

    return total_loss / n_batches, total_dist / n_batches, total_pck / n_batches


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    max_train = args.smoke_samples if args.smoke_test else args.max_train_samples
    max_val = max(8, args.smoke_samples // 4) if args.smoke_test else args.max_val_samples

    train_ds = CourtKeypointsDataset(args.train_json, args.images_dir, args.out_size, max_samples=max_train)
    val_ds = CourtKeypointsDataset(args.val_json, args.images_dir, args.out_size, max_samples=max_val)
    print(f"[setup] train samples={len(train_ds)}, val samples={len(val_ds)}", flush=True)

    num_workers = 0 if args.smoke_test else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=(num_workers > 0))

    model = CourtKeypointNet(n_keypoints=N_KEYPOINTS, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    loss_fn = torch.nn.SmoothL1Loss()

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
    best_dist = float("inf")
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    last_ckpt_path = os.path.join(args.ckpt_dir, args.last_ckpt_name)

    sample_images, sample_kps = next(iter(train_loader))
    print(f"[smoke] image batch shape={tuple(sample_images.shape)}, kps batch shape={tuple(sample_kps.shape)}", flush=True)
    assert sample_kps.shape[1] == 2 * N_KEYPOINTS

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_dist, train_pck = run_epoch(
            model, train_loader, device, optimizer, scaler, loss_fn, args.pck_thresh_px, train=True)
        train_time = time.time() - t0

        t1 = time.time()
        val_loss, val_dist, val_pck = run_epoch(
            model, val_loader, device, optimizer, scaler, loss_fn, args.pck_thresh_px, train=False)
        val_time = time.time() - t1

        mem_str = ""
        if device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" | peak_alloc={alloc:.2f}GB peak_reserved={reserved:.2f}GB"

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.5f} train_dist={train_dist:.1f}px train_pck@{args.pck_thresh_px:.0f}px={train_pck:.3f} | "
            f"val_loss={val_loss:.5f} val_dist={val_dist:.1f}px val_pck@{args.pck_thresh_px:.0f}px={val_pck:.3f} | "
            f"train_time={train_time:.1f}s val_time={val_time:.1f}s{mem_str}",
            flush=True,
        )

        if not args.smoke_test and not args.no_checkpoint:
            ckpt_dict = {"model_state": model.state_dict(),
                         "optimizer_state": optimizer.state_dict(),
                         "scaler_state": scaler.state_dict(),
                         "args": vars(args),
                         "epoch": epoch, "val_dist": val_dist, "val_pck": val_pck}
            torch.save(ckpt_dict, last_ckpt_path)
            if val_dist < best_dist:
                best_dist = val_dist
                torch.save(ckpt_dict, ckpt_path)
                print(f"[checkpoint] new best val_dist={val_dist:.1f}px -> saved {ckpt_path}", flush=True)

    if args.smoke_test:
        print("[smoke] SMOKE TEST COMPLETE -- shapes OK, loop ran without error.", flush=True)
        if device.type == "cuda":
            print(f"[smoke] peak GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    else:
        print(f"[done] best val_dist={best_dist:.1f}px, checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
