"""
Export a trained YOLOv8-seg court model to ONNX.

Note: this Ultralytics version (8.3.252) rejects int8=True for format="onnx"
("argument 'int8' is not supported for format='onnx'") -- INT8 export for this
format appears to have been dropped/changed upstream. Exporting plain float32
ONNX instead; revisit INT8 (e.g. via a different target format, or a separate
post-export quantization pass) if edge-deployment size/latency becomes an issue.

Usage:
    python training/court_detection/export_onnx.py \
        --weights training/court_detection/runs/court_seg/weights/best.pt
"""
import argparse
import os

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, "data", "data.yaml")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=DATA_YAML)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    # simplify=False: onnxslim's simplification pass hits the same onnx==1.22.0 /
    # protobuf==5.29.5 native ABI crash as onnx.checker (see ball_tracking/export_onnx.py).
    # Not required for correctness -- the onnxruntime inference check below is.
    out_path = model.export(format="onnx", data=args.data, imgsz=args.imgsz, simplify=False)
    print(f"Exported ONNX model -> {out_path}")

    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: dummy})
    print(f"ONNX Runtime sanity check OK, output shapes={[o.shape for o in out]}")


if __name__ == "__main__":
    main()
