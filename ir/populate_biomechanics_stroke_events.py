"""
Builds real StrokeEvent examples WITH BiomechanicsInfo populated, across 4
different stroke classes, for the RAG layer's stroke-level coaching
validation (rag/DESIGN.md "Stroke-level biomechanics -> RAG integration").

Same COMPOSITE pattern already established and documented in ir/populate.py
(real ball-tracking + real court-keypoints + real stroke-classification
from three independently-real but unrelated source clips) -- see that
file's module docstring for the full rationale. The one addition here:
`stroke` AND `biomechanics` are derived from the SAME real THETIS pose
sequence (not two unrelated sources), so those two components are actually
self-consistent with each other, unlike the ball/court pairing.

Only uses `clean`-bucket clips (per ir/populate_biomechanics_full.py's
full-dataset run, ir/example_output/biomechanics_full_dataset.csv) -- i.e.
real, already-validated-at-scale biomechanics, not a clip that would have
been excluded for degenerate camera angle / arm extension.

Usage:
    python ir/populate_biomechanics_stroke_events.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.schema import StrokeEvent, ModelProvenance  # noqa: E402
from ir.biomechanics import extract_biomechanics_features  # noqa: E402
from ir.populate import (  # noqa: E402
    ball_dataset_mod, stroke_dataset_mod, build_example_stroke_event,
    REPO_ROOT as POPULATE_REPO_ROOT,
)
import json as _json  # noqa: E402

# Clean (non-excluded), full-dataset-verified test-split examples, one per
# class, picked from ir/example_output/biomechanics_full_dataset.csv.
CLEAN_TEST_EXAMPLES = {
    "forehand": 10,  # p22_foreopen_s1.avi -- StrokeGRU predicted_class matches true label
    "backhand": 32,  # p28_backhand2h_s2.avi -- StrokeGRU predicted_class matches true label
    "serve": 1,      # p52_serflat_s3.avi -- StrokeGRU predicted_class matches true label
    "volley": 2,     # p28_bvolley_s1.avi -- StrokeGRU predicted_class matches true label
}
# Note: row_idx=3 (forehand) and row_idx=5 (backhand) were the first clean
# examples found, but StrokeGRU (test macro-F1 = 0.834, not perfect)
# predicted "volley" and "forehand" respectively for those two -- a real
# misclassification, not a bug here. Swapped to indices where predicted_class
# matches the true label so this validation exercises 4 genuinely distinct
# predicted classes; the mismatch itself is flagged as a finding in
# rag/DESIGN.md (percentile ranking keys off stroke.predicted_class, which
# inherits the classifier's own error rate).


def attach_real_biomechanics(event: StrokeEvent, pose_seq, source_file: str) -> StrokeEvent:
    features = extract_biomechanics_features(pose_seq)
    from ir.schema import BiomechanicsInfo
    biomechanics = BiomechanicsInfo(
        **features,
        source_model=ModelProvenance(
            component="biomechanics",
            checkpoint="pose_landmarker_full.task (upstream MediaPipe Pose landmarks; "
                       "geometric derivation itself has no learned weights)",
            metric_name="n/a", metric_value=None,
            source_file=source_file,
        ),
    )
    return event.model_copy(update={"biomechanics": biomechanics})


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

    ball_court_pairs = [
        (ball_clips[30], court_entries[300]),
        (ball_clips[40], court_entries[350]),
        (ball_clips[50], court_entries[400]),
        (ball_clips[60], court_entries[450]),
    ]

    for (collapsed_class, row_idx), (ball_clip, court_entry) in zip(CLEAN_TEST_EXAMPLES.items(), ball_court_pairs):
        stroke_x, stroke_y_idx = stroke_test_ds[row_idx]
        stroke_label = stroke_dataset_mod.COLLAPSED_CLASSES[stroke_y_idx.item()]
        assert stroke_label == collapsed_class, f"expected {collapsed_class}, got {stroke_label} at row {row_idx}"

        source_file = f"StrokeClassification/processed/test/row_idx={row_idx} (collapsed class: {collapsed_class})"
        event = build_example_stroke_event(
            ball_clip=ball_clip, ball_ckpt=ball_ckpt,
            court_entry=court_entry, court_images_dir=court_images_dir, court_ckpt=court_ckpt,
            stroke_x=stroke_x, stroke_label=source_file, stroke_ckpt=stroke_ckpt,
            player_id=f"p_biomech_demo_{collapsed_class}", player_side="near",
        )
        pose_seq = stroke_x.numpy()
        event = attach_real_biomechanics(event, pose_seq, source_file)

        print(f"[{collapsed_class}] event_id={event.event_id} "
              f"predicted_class={event.stroke.predicted_class} "
              f"wrist_height_ratio={event.biomechanics.wrist_height_ratio:.3f}")

        out_path = os.path.join(out_dir, f"stroke_event_biomech_{collapsed_class}.json")
        with open(out_path, "w") as f:
            f.write(event.model_dump_json(indent=2))
        print(f"  wrote {out_path}")
