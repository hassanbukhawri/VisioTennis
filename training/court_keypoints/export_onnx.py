"""
Export a trained CourtKeypointNet checkpoint to ONNX.

Usage:
    python training/court_keypoints/export_onnx.py \
        --ckpt training/court_keypoints/checkpoints/court_kp_best.pt \
        --out training/court_keypoints/checkpoints/court_kp.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CourtKeypointNet, N_KEYPOINTS


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
    out_size = train_args["out_size"]

    model = CourtKeypointNet(n_keypoints=N_KEYPOINTS, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_path = args.out or args.ckpt.replace(".pt", ".onnx")
    dummy = torch.randn(1, 3, out_size, out_size)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["image"], output_names=["keypoints"],
        dynamic_axes={"image": {0: "batch"}, "keypoints": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX model (out_size={out_size}, n_keypoints={N_KEYPOINTS}, "
          f"trained val_dist={ckpt.get('val_dist')}) -> {out_path}")

    # Sanity-check via onnxruntime inference. onnx.checker.check_model() is skipped
    # -- it segfaults on this machine (onnx==1.22.0 / protobuf==5.29.5 native ABI
    # mismatch, unrelated to the exported model, see ball_tracking/export_onnx.py).
    # Running the graph through onnxruntime is the real end-to-end check anyway.
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": dummy.numpy().astype(np.float32)})
    print(f"ONNX Runtime sanity check OK, output shape={out[0].shape}")


if __name__ == "__main__":
    main()
