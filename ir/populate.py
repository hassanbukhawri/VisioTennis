"""
IR population code: constructs real StrokeEvent / SessionRecord instances
from real trained-model inference on real sample data.

IMPORTANT CAVEAT -- read before treating example_output/ as representative of
a coherent pipeline: no single dataset in this project spans ball-tracking +
court-keypoints + stroke-classification for the SAME real rally/stroke.
Ball Tracker+Bounce, Court_Detector_Keypoints, and StrokeClassification
(THETIS) are three independent, unrelated video sources. The example
StrokeEvents built here are therefore COMPOSITES: real trained-model
inference on three independently-real but unrelated source clips, stitched
into one illustrative record to demonstrate the schema validates against
genuine model output (not fabricated numbers). This is NOT a demonstration of
one coherent end-to-end run on a single video -- that needs continuous match
footage with ball/court/stroke ground truth attached to the same rally, which
this project doesn't have (see DESIGN.md decision #2, "Not yet built").
Each component's real source file is recorded in its ModelProvenance.source_file
so this compositing is traceable, not hidden.

Usage:
    python ir/populate.py
"""
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.schema import (  # noqa: E402
    StrokeEvent, BallInfo, BallFramePosition, PixelPoint, CourtPoint,
    CourtInfo, StrokeInfo, ModelProvenance, FrameRange, SessionRecord,
    STROKE_CLASSES, ANOMALY_FEATURE_COLUMNS,
)
from ir.geometry import compute_homography, pixel_to_court  # noqa: E402


def _load_module(unique_name, rel_path):
    """training/*/{model,dataset}.py all use generic filenames -- a plain
    sys.path-based import would silently collide across folders (Python
    caches modules by name, so a second `import model` after the first would
    return the FIRST folder's model.py). Load each by explicit file path
    under a unique name instead."""
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


ball_dataset_mod = _load_module("_ir_ball_dataset", "training/ball_tracking/dataset.py")
ball_model_mod = _load_module("_ir_ball_model", "training/ball_tracking/model.py")
court_dataset_mod = _load_module("_ir_court_dataset", "training/court_keypoints/dataset.py")
court_model_mod = _load_module("_ir_court_model", "training/court_keypoints/model.py")
stroke_dataset_mod = _load_module("_ir_stroke_dataset", "training/stroke_classification/dataset.py")
stroke_model_mod = _load_module("_ir_stroke_model", "training/stroke_classification/model.py")
anomaly_train_mod = _load_module("_ir_anomaly_train", "training/anomaly_detection/train.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BALL_NATIVE_W, BALL_NATIVE_H = ball_dataset_mod.NATIVE_W, ball_dataset_mod.NATIVE_H  # 1280, 720
COURT_NATIVE_W, COURT_NATIVE_H = court_dataset_mod.NATIVE_W, court_dataset_mod.NATIVE_H  # 1280, 720
assert (BALL_NATIVE_W, BALL_NATIVE_H) == (COURT_NATIVE_W, COURT_NATIVE_H), (
    "ball and court datasets assume different native resolutions -- "
    "the shared-pixel-space assumption behind homography-applied-to-ball breaks"
)


# ---------------------------------------------------------------------------
# Model 1A: real TrackNet inference on a real Ball Tracker+Bounce clip
# ---------------------------------------------------------------------------

def populate_ball_info(clip, checkpoint_path, n_frames=12, conf_thresh=0.5):
    """clip: a ClipRecord from ball_dataset_mod.discover_clips(). Runs TrackNet
    over non-overlapping 3-frame windows across the clip's first n_frames real
    frames -- genuine forward passes, not fabricated positions. Ground-truth
    event_type (hit/bounce) is read from clip.status (decision #3 -- this is
    ground truth, not a model prediction; see DESIGN.md for why)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    seq_len = ckpt["args"]["seq_len"]
    out_w, out_h = ckpt["args"]["out_w"], ckpt["args"]["out_h"]
    model = ball_model_mod.TrackNet(seq_len=seq_len, base_ch=32).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sx_native = BALL_NATIVE_W / out_w  # scale factor: TrackNet output-space -> native pixel space
    sy_native = BALL_NATIVE_H / out_h

    n_frames = min(n_frames, len(clip.frame_files))
    n_frames -= n_frames % seq_len  # keep whole windows only
    trajectory = []

    with torch.no_grad():
        for start in range(0, n_frames, seq_len):
            frames = []
            for i in range(start, start + seq_len):
                img = cv2.imread(clip.frame_files[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frames.append(img.astype(np.float32) / 255.0)
            stack = np.concatenate(frames, axis=-1).transpose(2, 0, 1)  # (3*seq_len, H, W)
            x = torch.from_numpy(np.ascontiguousarray(stack)).float().unsqueeze(0).to(DEVICE)

            heatmaps = model(x)[0]  # (seq_len, out_h, out_w)
            for t in range(seq_len):
                hm = heatmaps[t]
                peak_val, peak_idx = hm.view(-1).max(dim=0)
                py = int(peak_idx.item() // out_w)
                px = int(peak_idx.item() % out_w)
                frame_idx = start + t
                visible = float(peak_val.item()) >= conf_thresh

                x_native = px * sx_native
                y_native = py * sy_native
                raw_status = clip.status[frame_idx]  # ground truth, see docstring
                event_type = None
                if not np.isnan(raw_status):
                    event_type = {0.0: "flight", 1.0: "hit", 2.0: "bounce"}[float(raw_status)]

                trajectory.append(BallFramePosition(
                    frame=frame_idx,
                    pixel=PixelPoint(x_px=x_native, y_px=y_native),
                    court=None,  # filled in later once we have a homography (see build_example_stroke_event)
                    visible=visible,
                    confidence=min(max(float(peak_val.item()), 0.0), 1.0),
                    event_type=event_type,
                ))

    hit_frames = [p.frame for p in trajectory if p.event_type == "hit"]
    bounce_frames = [p.frame for p in trajectory if p.event_type == "bounce"]

    return BallInfo(
        trajectory=trajectory,
        contact_frame=hit_frames[0] if hit_frames else None,
        bounce_frame=bounce_frames[0] if bounce_frames else None,
        source_model=ModelProvenance(
            component="ball", checkpoint=os.path.basename(checkpoint_path),
            metric_name="val_pck@4px", metric_value=ckpt.get("val_pck", float("nan")),
            source_file=f"{clip.game}/{clip.clip}",
        ),
    ), trajectory


# ---------------------------------------------------------------------------
# Model 1B: real CourtKeypointNet inference + real homography
# ---------------------------------------------------------------------------

def populate_court_info(entry, images_dir, checkpoint_path):
    """entry: one raw {"id", "kps", ...} dict from data_train.json / data_val.json.
    Runs CourtKeypointNet on the real image, computes a real homography from
    the predicted (not ground-truth) keypoints via cv2.findHomography."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    out_size = ckpt["args"]["out_size"]
    model = court_model_mod.CourtKeypointNet(n_keypoints=14, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_path = os.path.join(images_dir, entry["id"] + ".png")
    img_raw = cv2.imread(img_path)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - court_dataset_mod.IMAGENET_MEAN) / court_dataset_mod.IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()  # (28,) normalized [0,1]

    kps_norm = pred.reshape(14, 2)
    kps_native = kps_norm * np.array([COURT_NATIVE_W, COURT_NATIVE_H])  # -> native 1280x720 pixels
    keypoints_px = [PixelPoint(x_px=float(x_), y_px=float(y_)) for x_, y_ in kps_native]

    H, reproj_err = compute_homography(kps_native.tolist())

    court_info = CourtInfo(
        keypoints_px=keypoints_px,
        homography_matrix=H.tolist(),
        homography_reprojection_error_m=reproj_err,
        source_model=ModelProvenance(
            component="court", checkpoint=os.path.basename(checkpoint_path),
            metric_name="val_dist_px", metric_value=ckpt.get("val_dist", float("nan")),
            source_file=entry["id"] + ".png",
        ),
    )
    return court_info, H


# ---------------------------------------------------------------------------
# Model 2: real StrokeGRU inference on a real THETIS test clip
# ---------------------------------------------------------------------------

def populate_stroke_info(x_sequence, checkpoint_path, source_file=None):
    """x_sequence: (60, 99) float32 tensor, one real StrokeClassification test sample."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = stroke_model_mod.StrokeGRU(
        input_dim=99, hidden_dim=ckpt["args"]["hidden_dim"], num_layers=ckpt["args"]["num_layers"],
        num_classes=len(STROKE_CLASSES), dropout=ckpt["args"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(x_sequence.unsqueeze(0).to(DEVICE))[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    class_probabilities = {c: float(p) for c, p in zip(STROKE_CLASSES, probs)}
    predicted_class = max(class_probabilities, key=class_probabilities.get)

    return StrokeInfo(
        predicted_class=predicted_class,
        confidence=class_probabilities[predicted_class],
        class_probabilities=class_probabilities,
        source_model=ModelProvenance(
            component="stroke", checkpoint=os.path.basename(checkpoint_path),
            metric_name="val_macro_f1", metric_value=ckpt.get("val_macro_f1", float("nan")),
            source_file=source_file,
        ),
    )


# ---------------------------------------------------------------------------
# Composite StrokeEvent builder
# ---------------------------------------------------------------------------

def build_example_stroke_event(ball_clip, ball_ckpt, court_entry, court_images_dir, court_ckpt,
                                stroke_x, stroke_label, stroke_ckpt, player_id, player_side):
    ball_info, trajectory = populate_ball_info(ball_clip, ball_ckpt)
    court_info, H = populate_court_info(court_entry, court_images_dir, court_ckpt)
    stroke_info = populate_stroke_info(stroke_x, stroke_ckpt, source_file=stroke_label)

    # Project the ball trajectory (from the UNRELATED ball-tracking clip) through
    # THIS court's homography, purely to demonstrate the geometry pipeline runs
    # end-to-end -- see module docstring: these are different source videos, so
    # this court-position is illustrative of the transform, not a claim that this
    # ball was actually on this court.
    for i, p in enumerate(ball_info.trajectory):
        x_m, y_m = pixel_to_court(H, p.pixel.x_px, p.pixel.y_px)
        ball_info.trajectory[i] = p.model_copy(update={"court": CourtPoint(x_m=x_m, y_m=y_m)})

    event = StrokeEvent(
        player_id=player_id,
        player_side=player_side,
        source_video=(
            f"COMPOSITE (not one real video): ball={ball_clip.game}/{ball_clip.clip}, "
            f"court={court_entry['id']}, stroke={stroke_label}"
        ),
        frame_range=FrameRange(start_frame=0, end_frame=len(ball_info.trajectory) - 1, fps=30.0),
        ball=ball_info,
        court=court_info,
        stroke=stroke_info,
    )
    return event


# ---------------------------------------------------------------------------
# Model 4: SessionRecord from real IsolationForest fit on demo (synthetic) data
# ---------------------------------------------------------------------------

def populate_session_records(n_examples=3, random_state=42):
    """No real session dataset exists (see anomaly_detection/README / PROJECT_LOG),
    so this necessarily reuses the same clearly-synthetic demo data as
    `train.py --demo`. The IsolationForest fit/predict/explain below is real
    code executing for real, not fabricated -- only the INPUT is synthetic."""
    df = anomaly_train_mod.make_synthetic_demo_data(random_state=random_state)
    X = df[list(ANOMALY_FEATURE_COLUMNS)].values.astype(float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0

    from sklearn.ensemble import IsolationForest
    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=random_state, n_jobs=-1)
    model.fit(X)
    flags = model.predict(X)
    scores = model.decision_function(X)

    flagged_idx = np.where(flags == -1)[0][:n_examples]
    records = []
    for i in flagged_idx:
        row = df.iloc[i]
        explanation = anomaly_train_mod.explain_anomaly(row, mean, std, list(ANOMALY_FEATURE_COLUMNS), top_k=3)
        records.append(SessionRecord(
            player_id=row["player_id"],
            features={c: float(row[c]) for c in ANOMALY_FEATURE_COLUMNS},
            anomaly_flag=int(flags[i]),
            anomaly_score=float(scores[i]),
            explanation=explanation,
            source_model=ModelProvenance(
                component="anomaly", checkpoint="IsolationForest (fit at population time, not persisted)",
                metric_name="contamination", metric_value=0.1,
                source_file="synthetic_demo_data (no real session dataset exists, see DESIGN.md)",
            ),
        ))
    return records


if __name__ == "__main__":
    out_dir = os.path.join(REPO_ROOT, "ir", "example_output")
    os.makedirs(out_dir, exist_ok=True)

    ball_clips = ball_dataset_mod.discover_clips()
    with open(os.path.join(REPO_ROOT, "data/VisioTennis/Dataset/Court_Detector_Keypoints/data/data_val.json")) as f:
        court_entries = json.load(f)
    court_images_dir = os.path.join(
        REPO_ROOT, "data/VisioTennis/Dataset/Court_Detector_Keypoints/data/images")
    stroke_test_ds = stroke_dataset_mod.StrokeDataset("test")

    ball_ckpt = os.path.join(REPO_ROOT, "training/ball_tracking/checkpoints/tracknet_best.pt")
    court_ckpt = os.path.join(REPO_ROOT, "training/court_keypoints/checkpoints/court_kp_best.pt")
    stroke_ckpt = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/stroke_gru_best.pt")

    examples = [
        dict(ball_clip=ball_clips[0], court_entry=court_entries[0], stroke_idx=0,
             player_id="p_demo_01", player_side="near"),
        dict(ball_clip=ball_clips[10], court_entry=court_entries[50], stroke_idx=100,
             player_id="p_demo_02", player_side="far"),
        dict(ball_clip=ball_clips[20], court_entry=court_entries[200], stroke_idx=250,
             player_id="p_demo_03", player_side="near"),
    ]

    print(f"[setup] device={DEVICE}")
    stroke_events = []
    for i, ex in enumerate(examples):
        stroke_x, stroke_y_idx = stroke_test_ds[ex["stroke_idx"]]
        stroke_label = stroke_dataset_mod.COLLAPSED_CLASSES[stroke_y_idx.item()]
        event = build_example_stroke_event(
            ball_clip=ex["ball_clip"], ball_ckpt=ball_ckpt,
            court_entry=ex["court_entry"], court_images_dir=court_images_dir, court_ckpt=court_ckpt,
            stroke_x=stroke_x, stroke_label=f"test_idx={ex['stroke_idx']} (true label: {stroke_label})",
            stroke_ckpt=stroke_ckpt,
            player_id=ex["player_id"], player_side=ex["player_side"],
        )
        stroke_events.append(event)
        out_path = os.path.join(out_dir, f"stroke_event_{i}.json")
        with open(out_path, "w") as f:
            f.write(event.model_dump_json(indent=2))
        print(f"[{i}] wrote {out_path}")
        print(f"    predicted_class={event.stroke.predicted_class} (true label: {stroke_label}) "
              f"confidence={event.stroke.confidence:.3f}")
        print(f"    homography_reprojection_error_m={event.court.homography_reprojection_error_m:.4f}")
        print(f"    ball trajectory: {len(event.ball.trajectory)} points, "
              f"contact_frame={event.ball.contact_frame}, bounce_frame={event.ball.bounce_frame}")

    print()
    session_records = populate_session_records()
    for i, rec in enumerate(session_records):
        out_path = os.path.join(out_dir, f"session_record_{i}.json")
        with open(out_path, "w") as f:
            f.write(rec.model_dump_json(indent=2))
        print(f"[session {i}] wrote {out_path} -- player={rec.player_id} flag={rec.anomaly_flag} "
              f"score={rec.anomaly_score:.3f}")
        print(f"    {rec.explanation}")

    print(f"\n[done] {len(stroke_events)} StrokeEvent + {len(session_records)} SessionRecord "
          f"example files written to {out_dir}")
