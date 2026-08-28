You are an expert Senior Machine Learning and Deep Learning Engineer working directly inside my project repository. I am developing "VisioTennis," an AI-first tennis performance analysis platform, as a Final Year Project. I need you to write, run, debug, and validate production-grade Python training scripts for the core CV/ML models in the pipeline — one model at a time, smoke-tested on my actual data before any full training run.

## Environment
- Local machine, GPU: NVIDIA RTX 3060 (assume 12GB VRAM unless `nvidia-smi` shows otherwise — check first)
- Frameworks: PyTorch (CUDA), Ultralytics, Scikit-learn
- Before writing any training code: run `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` to confirm the environment is correctly configured. Fix the environment first if it isn't.
- Use `torch.cuda.amp` (mixed precision) for all PyTorch training loops to fit comfortably in 12GB.
- For every model below: first write the data loader and run a 1–2 epoch smoke test on a small subset (e.g. 50–100 samples) to confirm shapes, loss direction, and GPU memory usage are sane — only then proceed to a full training run.
- Any full training run expected to take more than ~10 minutes must be launched as a background process (e.g. `nohup python train.py > logs/train.log 2>&1 &`), not run in the foreground. Check progress by tailing the log file, not by blocking on the process.
- Do not fabricate or assume dataset schemas. Inspect the actual files (open a sample `Label.csv`, a sample annotation JSON, a sample YOLO `.txt` label) before writing the `Dataset` class for that model, and write the loader against what you actually find.

## Repository dataset layout (confirmed present on disk)
```
data/VisioTennis/Dataset/
├── Ball Tracker+Bounce/
│   └── game{1..10}/
│       └── Clip{N}/
│           ├── 0000.jpg, 0001.jpg, ... (sequential frames)
│           └── Label.csv   (inspect columns before use — likely frame#, visibility, x, y, status)
├── Court_Detector/                     (Roboflow export, YOLO bbox format)
│   ├── train/images/*.jpg, train/labels/*.txt
│   ├── valid/images/*.jpg, valid/labels/*.txt
│   ├── test/images/*.jpg,  test/labels/*.txt
│   └── data.yaml
├── Court_Detector_Keypoints/
│   └── data/
│       ├── data_train.json, data_val.json  (14 court keypoints per image)
│       └── images/*.png
├── Pose_Estimation/
│   └── .../annotations/{backhand,forehand,ready_position,serve}.json
│   └── .../images/{backhand,forehand,ready_position,serve}/*.jpeg
└── StrokeClassification/processed/
    ├── class_map.csv
    ├── train/X_train.npy, labels_train.csv
    ├── val/X_val.npy, labels_val.csv
    └── test/X_test.npy, labels_test.csv
```

Note the known gaps — do not silently paper over these, flag them back to me:
- No player bounding-box annotations exist anywhere in this dataset. Do not train a player detector. Use a COCO-pretrained YOLOv8 (`person` class) as-is for player detection.
- Pose_Estimation currently has 4 stroke classes (backhand, forehand, ready_position, serve) — the target is 5 (add volley, smash). Flag this gap rather than inventing data.
- `StrokeClassification/processed/` already contains extracted `.npy` feature arrays — inspect whether these are raw pixels, optical-flow features, or pose-keypoint features before assuming what Model 2 should consume. This may let us skip a feature-extraction step entirely.

---

## MODEL 1A: Ball Tracking (TrackNet-style heatmap regression)

- **Task:** Track the tennis ball across frames, robust to motion blur and partial occlusion.
- **Architecture:** TrackNet-style — input a stack of N consecutive RGB frames (commonly 3), output a Gaussian heatmap over ball position per frame (not a bounding box).
- **Dataset:** `data/VisioTennis/Dataset/Ball Tracker+Bounce/game{1..10}/Clip{N}/` — sequential JPGs + `Label.csv` per clip.
- **Requirements:**
  1. First, inspect 2–3 `Label.csv` files and print their columns/sample rows so we confirm the schema before writing the `Dataset` class.
  2. Write a PyTorch `Dataset` that loads N-frame stacks and generates target Gaussian heatmaps centered on the labeled ball (x, y) position, handling frames with `visibility == 0` (no ball visible) correctly (e.g. all-zero heatmap or a mask).
  3. Define the TrackNet CNN architecture (encoder-decoder / heatmap regression head).
  4. Training loop: weighted loss (heatmap regression, e.g. weighted BCE or MSE), validation split by `game`, not by frame (avoid leakage between clips of the same game/rally).
  5. Evaluation: report distance-based accuracy (e.g. % of frames where predicted peak is within N pixels of ground truth) in addition to loss.
  6. Save best checkpoint by validation metric; export to ONNX.

## MODEL 1B: Court Keypoint / Line Detection (YOLOv8 + separate keypoint model)

- **Task:** Detect the court, either as bounding-box court markings (Roboflow set) or as 14 court keypoints (keypoints set) for homography/bird's-eye mapping.
- **Dataset A (bbox):** `Dataset/Court_Detector/{train,valid,test}/` — standard Roboflow YOLO export with `data.yaml`.
- **Dataset B (keypoints):** `Dataset/Court_Detector_Keypoints/data/` — `data_train.json` / `data_val.json` + PNGs.
- **Requirements:**
  1. Write a standard Ultralytics YOLOv8 fine-tuning script against `data.yaml` for the bbox court dataset (epochs=100, imgsz=640, device=0, batch tuned to fit 12GB — start at 16 and adjust after the smoke test).
  2. Separately, inspect `data_train.json` to confirm the keypoint schema, then write a lightweight keypoint-heatmap or direct-regression model (this does not need to be YOLOv8-pose unless you confirm the JSON is already in a YOLO-pose-compatible format — check before assuming).
  3. Export the YOLOv8 court model to INT8-quantized ONNX for edge deployment.
  4. Do not combine ball tracking and court detection into a single YOLOv8 job — they are separate models against separate datasets.

## MODEL 2: Stroke Classification

- **Task:** Classify strokes into forehand, backhand, serve, volley, smash (currently only 4 classes available — flag the missing 2 before training).
- **Primary approach — inspect first:** Check `StrokeClassification/processed/X_train.npy` shape and dtype. If these are already pose-keypoint sequences, build a sequence classifier (e.g. small GRU/LSTM or 1D-CNN over keypoint time series) directly on them — this is far cheaper on a 3060 than full two-stream video.
- **Fallback approach (only if the above data isn't usable as-is):** Two-Stream CNN using `Pose_Estimation` images.
  - Spatial stream: ResNet backbone on RGB frames.
  - Temporal stream: small CNN on **precomputed** optical flow (Farneback, computed and cached to disk in a separate offline preprocessing script — never compute optical flow inside the training loop/dataloader on this GPU).
  - Late fusion before the final FC classification layer.
- **Requirements:**
  1. Confirm which approach is viable before writing full training code.
  2. Custom PyTorch `Dataset` matching whichever approach is used.
  3. Training/validation loop with per-class F1 reported (not just overall accuracy) — this dataset is likely class-imbalanced.
  4. Checkpointing on best macro-F1.

## MODEL 3: Momentum Tracking (LSTM) — SCOPE ONLY, DO NOT BUILD YET

- Point-by-point match data (Sackmann / Match Charting Project / SCORE Network) is not yet present in the local repository.
- Do not write a training script against fabricated or placeholder data.
- Instead: write the `TennisMomentumLSTM` **model definition only** (architecture, forward pass, expected input tensor shape/feature spec) plus a short data-acquisition checklist describing exactly what files/format are needed before this can be trained. No training loop yet.

## MODEL 4: Anomaly / Fatigue Detection (Isolation Forest)

- **Task:** Flag anomalous player sessions from multi-dimensional performance vectors (first-serve %, court coverage speed, rally length, etc.).
- **Requirements:**
  1. This is CPU-only (sklearn) — no GPU/ONNX concerns.
  2. Write the `IsolationForest` training script against a placeholder feature-vector schema (list the expected columns explicitly in a comment/docstring), since no historical session dataset exists locally yet.
  3. Output both `-1/1` anomaly flags and continuous `decision_function` scores, plus a simple example of turning a flag into a human-readable explanation (which feature(s) deviated most).

---

## Process instructions
- Work through Model 1A first (cleanest, most complete dataset), fully smoke-tested and running, before moving to 1B, then 2, then 4. Model 3 is definitions-only for now.
- After each model's smoke test passes, tell me the observed epoch time and GPU memory usage before starting a full run, so I can decide whether to adjust batch size/epochs.
- Keep each model in its own script/folder (e.g. `training/ball_tracking/`, `training/court_detection/`, `training/stroke_classification/`, `training/anomaly_detection/`) with its own `requirements.txt` if dependencies differ.
