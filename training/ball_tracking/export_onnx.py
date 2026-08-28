"""
Export a trained TrackNet checkpoint to ONNX.

Usage:
    python training/ball_tracking/export_onnx.py \
        --ckpt training/ball_tracking/checkpoints/tracknet_best.pt \
        --out training/ball_tracking/checkpoints/tracknet.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TrackNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]
    seq_len, out_w, out_h = train_args["seq_len"], train_args["out_w"], train_args["out_h"]

    model = TrackNet(seq_len=seq_len, base_ch=32)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_path = args.out or args.ckpt.replace(".pt", ".onnx")
    dummy = torch.randn(1, 3 * seq_len, out_h, out_w)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["frames"], output_names=["heatmaps"],
        dynamic_axes={"frames": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX model (seq_len={seq_len}, input {out_w}x{out_h}, "
          f"trained val_pck={ckpt.get('val_pck')}) -> {out_path}")

    # Sanity-check the export loads and runs. Note: onnx.checker.check_model()
    # is skipped here -- it segfaults on this machine due to an onnx/protobuf
    # native ABI mismatch (onnx==1.22.0 checker vs protobuf==5.29.5), unrelated
    # to the exported model itself (onnx.load() alone parses fine). Running the
    # graph through onnxruntime is the real end-to-end check anyway.
    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx.load(out_path)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"frames": dummy.numpy().astype(np.float32)})
    print(f"ONNX Runtime sanity check OK, output shape={out[0].shape}")


if __name__ == "__main__":
    main()
