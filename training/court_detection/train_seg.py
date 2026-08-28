"""
YOLOv8-segmentation fine-tuning for court detection (Model 1B, Dataset A).

The Roboflow export is polygon segmentation labels (not bboxes -- see
prepare_data.py docstring), so we train yolov8*-seg rather than a plain detector.
Run prepare_data.py first to build the Ultralytics-conventional data/ layout.

Usage:
    python training/court_detection/prepare_data.py     # one-time
    python training/court_detection/train_seg.py --smoke-test
    python training/court_detection/train_seg.py --epochs 100 --batch 16
"""
import argparse
import os

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, "data", "data.yaml")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_YAML)
    p.add_argument("--model", default="yolov8n-seg.pt", help="base checkpoint to fine-tune")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=os.path.join(SCRIPT_DIR, "runs"))
    p.add_argument("--name", default="court_seg")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"{args.data} not found -- run prepare_data.py first")

    model = YOLO(args.model)

    if args.smoke_test:
        print("[smoke] running 1 epoch at reduced imgsz/batch to sanity-check the pipeline")
        results = model.train(
            data=args.data, epochs=1, imgsz=320, batch=4, device=args.device,
            project=args.project, name=args.name + "_smoke", exist_ok=True,
            verbose=True, plots=False,
        )
        print("[smoke] SMOKE TEST COMPLETE -- training loop ran without error.")
        print(f"[smoke] results saved to {results.save_dir}")
        return

    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name, exist_ok=True,
    )


if __name__ == "__main__":
    main()
