"""
Clean, player-disjoint re-split of the pose-based StrokeGRU dataset --
the retroactive fix for the leakage found while building Path B's data prep
(PROJECT_LOG.md, 2026-08-20: "existing StrokeGRU model's original train/val/
test split has 100% player overlap across all splits"). Same class of bug as
the original court-detection leaky-split fix, now getting the same
clean-split retrain-and-compare treatment, per explicit instruction.

Does NOT re-run MediaPipe pose extraction -- the existing processed/{split}/
X_{split}.npy + labels_{split}.csv already contain all 1980 clips' extracted
99-dim pose sequences, just assigned to splits without regard to player
identity. This script merges all 3 existing splits back into one 1980-clip
pool (data itself is fine; only the split assignment was leaky), then
re-assigns players to train/val/test using the EXACT SAME splitting function
already built and verified for Path B (`prepare_video_split.split_players`,
SPLIT_SEED=42, 39/8/8 players) -- imported directly, not reimplemented, so
this is guaranteed to be the identical split logic, not just a similar one.

Usage:
    python training/stroke_classification/prepare_clean_pose_split.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from prepare_video_split import split_players, verify_no_leakage

OLD_ROOT = os.path.join("data", "VisioTennis", "Dataset", "StrokeClassification", "processed")
NEW_ROOT = os.path.join("data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")

FILENAME_RE = re.compile(r"^(p\d+)_.*_s\d+\.avi$")


def load_and_merge():
    Xs, labels_frames = [], []
    for split in ("train", "val", "test"):
        X = np.load(os.path.join(OLD_ROOT, split, f"X_{split}.npy"))
        labels = pd.read_csv(os.path.join(OLD_ROOT, split, f"labels_{split}.csv"))
        assert X.shape[0] == len(labels)
        Xs.append(X)
        labels_frames.append(labels)

    X_all = np.concatenate(Xs, axis=0)
    labels_all = pd.concat(labels_frames, ignore_index=True)
    assert X_all.shape[0] == len(labels_all) == 1980, \
        f"expected 1980 merged clips, got X={X_all.shape[0]} labels={len(labels_all)}"

    player_ids = []
    for fname in labels_all["filename"]:
        m = FILENAME_RE.match(fname)
        assert m, f"unexpected filename pattern: {fname}"
        player_ids.append(m.group(1))
    labels_all["player_id"] = player_ids

    return X_all, labels_all


def main():
    X_all, labels_all = load_and_merge()
    print(f"[setup] merged {X_all.shape[0]} clips from the 3 existing (leaky) splits, "
          f"X shape={X_all.shape}")
    print(f"[setup] {labels_all['player_id'].nunique()} unique players")

    # Reuse Path B's exact split_players() -- it only reads element [3] of each
    # tuple (player_id), so the other 3 fields are irrelevant filler here.
    fake_clips = [(None, None, None, pid) for pid in labels_all["player_id"]]
    player_split = split_players(fake_clips)
    for split, players in player_split.items():
        print(f"[split] {split}: {len(players)} players")

    os.makedirs(NEW_ROOT, exist_ok=True)
    rows_by_split = {"train": [], "val": [], "test": []}
    for split, players in player_split.items():
        mask = labels_all["player_id"].isin(players)
        idx = np.where(mask.values)[0]
        X_split = X_all[idx]
        labels_split = labels_all.iloc[idx][["filename", "label", "label_idx"]].reset_index(drop=True)

        out_dir = os.path.join(NEW_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"X_{split}.npy"), X_split)
        labels_split.to_csv(os.path.join(out_dir, f"labels_{split}.csv"), index=False)
        print(f"[write] {split}: {X_split.shape[0]} clips -> {out_dir}")

        # verify_no_leakage indexes filepath at [0] and player_id at [4] (it expects
        # the 5-tuple shape write_manifests() produces) -- use filename as the
        # dedup key here since there's no filesystem path in this pose dataset.
        rows_by_split[split] = [
            (fname, None, None, None, pid)
            for fname, pid in zip(labels_all.iloc[idx]["filename"], labels_all.iloc[idx]["player_id"])
        ]

    verify_no_leakage(rows_by_split)

    print("\n[class balance] collapsed-native label counts per split:")
    for split in ("train", "val", "test"):
        labels_split = pd.read_csv(os.path.join(NEW_ROOT, split, f"labels_{split}.csv"))
        print(f"  {split}: {len(labels_split)} clips, "
              f"{labels_split['label_idx'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
