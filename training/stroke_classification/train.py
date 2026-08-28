"""
Training script for VisioTennis Model 2: stroke classification (collapsed 5-class).

Usage:
    python training/stroke_classification/train.py --smoke-test
    python training/stroke_classification/train.py --epochs 40 --batch-size 64
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import StrokeDataset, COLLAPSED_CLASSES, DATA_ROOT
from model import StrokeGRU

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DATA_ROOT)  # override to point at a different processed/{split} tree, e.g. processed_clean for the player-disjoint re-split
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)  # dataset is tiny, IPC overhead not worth it

    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-samples", type=int, default=80)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--ckpt-name", default="stroke_gru_best.pt")
    p.add_argument("--last-ckpt-name", default="stroke_gru_last.pt")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def run_epoch(model, loader, device, optimizer, scaler, loss_fn, train):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(y.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    macro_f1 = f1_score(targets, preds, average="macro", labels=list(range(len(COLLAPSED_CLASSES))), zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None, labels=list(range(len(COLLAPSED_CLASSES))), zero_division=0)
    acc = float((preds == targets).mean())
    return total_loss / max(n_batches, 1), macro_f1, per_class_f1, acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    max_train = args.smoke_samples if args.smoke_test else args.max_train_samples
    max_val = max(8, args.smoke_samples // 4) if args.smoke_test else args.max_val_samples

    train_ds = StrokeDataset("train", data_root=args.data_root, max_samples=max_train)
    val_ds = StrokeDataset("val", data_root=args.data_root, max_samples=max_val)
    print(f"[setup] data_root={args.data_root}", flush=True)
    print(f"[setup] train samples={len(train_ds)} class_counts={train_ds.class_counts().tolist()}", flush=True)
    print(f"[setup] val samples={len(val_ds)} class_counts={val_ds.class_counts().tolist()}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = StrokeGRU(input_dim=99, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                       num_classes=len(COLLAPSED_CLASSES), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # inverse-frequency class weights -- smash is a real 3:1 minority vs forehand/backhand/serve
    counts = train_ds.class_counts().float()
    class_weights = (counts.sum() / (len(counts) * counts)).to(device)
    print(f"[setup] class weights (inverse frequency) = {class_weights.tolist()}", flush=True)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    if args.resume:
        print(f"[setup] resuming from {args.resume}", flush=True)
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            print("[setup] restored optimizer state", flush=True)
        if "scaler_state" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler_state"])

    epochs = 2 if args.smoke_test else args.epochs
    best_macro_f1 = -1.0
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    last_ckpt_path = os.path.join(args.ckpt_dir, args.last_ckpt_name)

    sample_x, sample_y = next(iter(train_loader))
    print(f"[smoke] batch x shape={tuple(sample_x.shape)}, y shape={tuple(sample_y.shape)}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_f1, train_per_class, train_acc = run_epoch(
            model, train_loader, device, optimizer, scaler, loss_fn, train=True)
        train_time = time.time() - t0

        t1 = time.time()
        val_loss, val_f1, val_per_class, val_acc = run_epoch(
            model, val_loader, device, optimizer, scaler, loss_fn, train=False)
        val_time = time.time() - t1

        mem_str = ""
        if device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            mem_str = f" | peak_alloc={alloc:.2f}GB peak_reserved={reserved:.2f}GB"

        per_class_str = " ".join(f"{c}={f:.3f}" for c, f in zip(COLLAPSED_CLASSES, val_per_class))
        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} train_macroF1={train_f1:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macroF1={val_f1:.3f} | "
            f"val_per_class_F1[{per_class_str}] | "
            f"train_time={train_time:.1f}s val_time={val_time:.1f}s{mem_str}",
            flush=True,
        )

        if not args.smoke_test and not args.no_checkpoint:
            ckpt_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "args": vars(args),
                "epoch": epoch, "val_macro_f1": val_f1, "val_loss": val_loss,
                "val_per_class_f1": {c: float(f) for c, f in zip(COLLAPSED_CLASSES, val_per_class)},
            }
            torch.save(ckpt_dict, last_ckpt_path)
            if val_f1 > best_macro_f1:
                best_macro_f1 = val_f1
                torch.save(ckpt_dict, ckpt_path)
                print(f"[checkpoint] new best val_macroF1={val_f1:.3f} -> saved {ckpt_path}", flush=True)

    if args.smoke_test:
        print("[smoke] SMOKE TEST COMPLETE -- shapes OK, loop ran without error.", flush=True)
        if device.type == "cuda":
            print(f"[smoke] peak GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    else:
        print(f"[done] best val_macroF1={best_macro_f1:.3f}, checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
