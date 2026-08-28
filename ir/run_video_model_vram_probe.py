"""
VRAM/hardware feasibility probe for Path B (video-based fine-tuning
candidates) -- NOT a training script. Runs exactly one forward + backward
pass per candidate model at a plausible small fine-tuning batch size, on
dummy input (no dataset, no optimizer step loop, nothing saved), and reads
real torch.cuda.max_memory_allocated(). This measures a genuine floor: real
training also needs to hold the full dataset/dataloader pipeline in
memory/CPU and activation memory scales with whatever batch size/clip length
is finally chosen -- treat these numbers as "confirmed lower bound on what a
real fine-tuning run needs," not a promise of the exact final config.

Each candidate is wrapped with a small NUM_CLASSES-output adapter layer on
top of its native pretrained output (rather than surgically replacing each
library's internal head, which has a different attribute path per library
and isn't needed for a memory measurement -- the adapter's own parameter
count is negligible next to the backbone, so this doesn't materially change
the measured footprint vs. a "real" head-replacement fine-tune).

Run this WHILE a real qwen3:8b Ollama instance is actually loaded (not just
documented as loaded elsewhere) so the concurrent-use headroom number is
measured against real usage, not reused blind from an earlier session.

Usage:
    python ir/run_video_model_vram_probe.py
"""
import gc
import sys
import time

import torch
import torch.nn as nn

BATCH_SIZE = 4  # a realistic small fine-tuning batch given limited real labeled data (see feasibility report item 4)
NUM_CLASSES = 5  # matches this project's collapsed stroke taxonomy


class AdapterHead(nn.Module):
    """Wraps a pretrained video model, adding a small Linear adapter on top
    of its native output -- see module docstring for why this is used
    instead of surgically replacing each library's internal head."""
    def __init__(self, backbone, native_out_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Linear(native_out_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        return self.adapter(feats)


def _measure(name, build_fn, input_shape, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    try:
        model = build_fn().to(device)
        model.train()
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(*input_shape, device=device)
        y = torch.randint(0, NUM_CLASSES, (input_shape[0],), device=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # One warmup step (cuDNN autotune/lazy-init), not counted -- then time
        # N real forward+backward+step iterations for a genuine per-batch
        # wall-clock estimate, not a guessed number.
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        n_timed = 10
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n_timed):
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize(device)
        s_per_batch = (time.perf_counter() - t0) / n_timed

        peak_bytes = torch.cuda.max_memory_allocated(device)
        print(f"[{name}] params={params / 1e6:.2f}M input_shape={tuple(input_shape)} "
              f"peak_vram={peak_bytes / (1024 ** 3):.3f} GB "
              f"s_per_batch={s_per_batch:.3f}s (batch={input_shape[0]}, {input_shape[0] / s_per_batch:.1f} clips/s)")
        del model, x, y, optimizer, out, loss
    except torch.cuda.OutOfMemoryError as e:
        print(f"[{name}] OOM at batch_size={input_shape[0]}: {str(e)[:200]}")
    except Exception as e:
        print(f"[{name}] FAILED to build/run: {type(e).__name__}: {str(e)[:300]}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No CUDA device available -- this probe is meaningless on CPU for a GPU feasibility question.")
        sys.exit(1)

    free_before, total = torch.cuda.mem_get_info(device)
    print(f"[setup] device={torch.cuda.get_device_name(device)} "
          f"free={free_before / (1024 ** 3):.2f}GB / total={total / (1024 ** 3):.2f}GB "
          f"(reflects whatever else is ALREADY loaded on the GPU right now, e.g. Ollama)\n")

    # -- torchvision r2plus1d_18 (already installed, no new dependency) --
    import torchvision.models.video as video_models
    _measure(
        "r2plus1d_18",
        lambda: AdapterHead(video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.DEFAULT), 400, NUM_CLASSES),
        input_shape=(BATCH_SIZE, 3, 16, 112, 112),  # (B, C, T, H, W) -- standard torchvision video shape
        device=device,
    )

    # -- X3D-XS (pytorchvideo) --
    try:
        import pytorchvideo.models.hub as hub
        _measure(
            "X3D-XS",
            lambda: AdapterHead(hub.x3d_xs(pretrained=True), 400, NUM_CLASSES),
            input_shape=(BATCH_SIZE, 3, 4, 182, 182),  # X3D-XS's own native 4-frame, 182x182 config
            device=device,
        )
    except ImportError:
        print("[X3D-XS] SKIPPED -- pytorchvideo not installed")

    # -- MoViNet-A0 (Atze00/MoViNet-pytorch, unofficial port) --
    try:
        from movinets import MoViNet
        from movinets.config import _C
        _measure(
            "MoViNet-A0",
            lambda: AdapterHead(MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=True), 600, NUM_CLASSES),
            input_shape=(BATCH_SIZE, 3, 8, 172, 172),  # a modest clip length/resolution for A0
            device=device,
        )
    except ImportError:
        print("[MoViNet-A0] SKIPPED -- MoViNet-pytorch not installed")

    free_after, _ = torch.cuda.mem_get_info(device)
    print(f"\n[done] free memory now: {free_after / (1024 ** 3):.2f}GB "
          f"(should be close to free_before if cleanup worked)")
