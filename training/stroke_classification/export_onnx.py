"""
Export a trained StrokeGRU checkpoint to ONNX.

Usage:
    python training/stroke_classification/export_onnx.py \
        --ckpt training/stroke_classification/checkpoints/stroke_gru_best.pt \
        --out training/stroke_classification/checkpoints/stroke_gru.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import COLLAPSED_CLASSES, N_KEYPOINT_FEATURES, SEQ_LEN
from model import StrokeGRU


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]

    model = StrokeGRU(input_dim=N_KEYPOINT_FEATURES, hidden_dim=train_args["hidden_dim"],
                       num_layers=train_args["num_layers"], num_classes=len(COLLAPSED_CLASSES),
                       dropout=train_args["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_path = args.out or args.ckpt.replace(".pt", ".onnx")
    dummy = torch.randn(1, SEQ_LEN, N_KEYPOINT_FEATURES)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["keypoint_seq"], output_names=["logits"],
        dynamic_axes={"keypoint_seq": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX model (seq_len={SEQ_LEN}, n_features={N_KEYPOINT_FEATURES}, "
          f"classes={COLLAPSED_CLASSES}, val_macro_f1={ckpt.get('val_macro_f1')}) -> {out_path}")

    # onnx.checker.check_model() is skipped -- segfaults on this machine (onnx==1.22.0 /
    # protobuf==5.29.5 native ABI mismatch, see ball_tracking/export_onnx.py). onnxruntime
    # inference below is the meaningful check.
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"keypoint_seq": dummy.numpy().astype(np.float32)})
    print(f"ONNX Runtime sanity check OK, output shape={out[0].shape}")


if __name__ == "__main__":
    main()
