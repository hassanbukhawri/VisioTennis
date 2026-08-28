"""
Video-level train/val/test split for Path B (r2plus1d_18 fine-tuning on raw
THETIS RGB footage), source: data/VisioTennis/Dataset/StrokeClassification/
raw_rgb/<native_class>/p<player>_<stroke>_s<shot>.avi (downloaded per the
2026-08-12 PROJECT_LOG.md entry -- 1980 files, 12 native classes x 165 each).

SPLIT AT THE PLAYER LEVEL, not the file level -- a stronger requirement than
"just don't split one video's frames across sets" (every raw_rgb/*.avi is
already a single atomic clip, one full stroke performance, so file-level
splitting alone would already prevent frame leakage). The reason to go
further: checking the EXISTING pose-based split this session
(data/VisioTennis/Dataset/StrokeClassification/processed/{train,val,test}/)
found all 55 players present in all 3 of ITS splits (100% player overlap,
train&test intersection == train's own player set) -- that split is safe
against frame leakage but not against subject-identity leakage: a model
could learn to recognize a player's body/clothing rather than the stroke.
This script fixes that for the video model by assigning each PLAYER's
entire clip set (all 12 classes x 3 shots = 36 clips) wholly to one split,
so no player's face/body/clothing is ever seen in both train and eval.

Every one of the 55 players contributes exactly 3 shots to every one of the
12 native classes (confirmed by direct inspection before writing this
script, zero exceptions) -- so a player-level split is automatically
class-balanced too, no stratification logic needed.

Reuses the EXACT native-class -> 5-class collapse already decided for the
pose-based model (training/stroke_classification/dataset.py:
NATIVE_TO_COLLAPSED), keyed by folder name here instead of label_idx, so
both models target the identical 5-class taxonomy.

Usage:
    python training/stroke_classification/prepare_video_split.py
"""
import csv
import os
import random
import re

RAW_ROOT = os.path.join("data", "VisioTennis", "Dataset", "StrokeClassification", "raw_rgb")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_splits")

COLLAPSED_CLASSES = ["forehand", "backhand", "serve", "volley", "smash"]
COLLAPSED_IDX = {c: i for i, c in enumerate(COLLAPSED_CLASSES)}

# native folder name -> collapsed class. Identical decision to dataset.py's
# NATIVE_TO_COLLAPSED (there keyed by label_idx 0-11 in class_map.csv order,
# which is alphabetical folder order -- same mapping, different key type).
NATIVE_TO_COLLAPSED = {
    "backhand": "backhand",
    "backhand2hands": "backhand",
    "backhand_slice": "backhand",
    "backhand_volley": "volley",
    "flat_service": "serve",
    "forehand_flat": "forehand",
    "forehand_openstands": "forehand",
    "forehand_slice": "forehand",
    "forehand_volley": "volley",
    "kick_service": "serve",
    "slice_service": "serve",
    "smash": "smash",
}

FILENAME_RE = re.compile(r"^(p\d+)_.*_s\d+\.avi$")

SPLIT_SEED = 42
# 55 players total -> 39 train / 8 val / 8 test (70.9% / 14.5% / 14.5%),
# close to this project's existing 70/15/15 convention (training/
# stroke_classification/dataset.py's processed split), chosen at the player
# level instead of the row level.
N_VAL_PLAYERS = 8
N_TEST_PLAYERS = 8


def collect_clips():
    assert set(NATIVE_TO_COLLAPSED) == set(os.listdir(RAW_ROOT)), \
        f"raw_rgb folder set doesn't match NATIVE_TO_COLLAPSED keys: {sorted(os.listdir(RAW_ROOT))}"

    clips = []  # (filepath, native_class, collapsed_class, player_id)
    for native_class in sorted(os.listdir(RAW_ROOT)):
        class_dir = os.path.join(RAW_ROOT, native_class)
        for fname in sorted(os.listdir(class_dir)):
            m = FILENAME_RE.match(fname)
            assert m, f"unexpected filename pattern: {native_class}/{fname}"
            player_id = m.group(1)
            clips.append((
                os.path.join(class_dir, fname),
                native_class,
                NATIVE_TO_COLLAPSED[native_class],
                player_id,
            ))
    return clips


def split_players(clips):
    players = sorted({c[3] for c in clips})
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(players)

    test_players = set(players[:N_TEST_PLAYERS])
    val_players = set(players[N_TEST_PLAYERS:N_TEST_PLAYERS + N_VAL_PLAYERS])
    train_players = set(players[N_TEST_PLAYERS + N_VAL_PLAYERS:])

    assert not (train_players & val_players)
    assert not (train_players & test_players)
    assert not (val_players & test_players)
    assert train_players | val_players | test_players == set(players)

    return {"train": train_players, "val": val_players, "test": test_players}


def write_manifests(clips, player_split):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows_by_split = {"train": [], "val": [], "test": []}

    for filepath, native_class, collapsed_class, player_id in clips:
        for split, players in player_split.items():
            if player_id in players:
                rows_by_split[split].append((filepath, native_class, collapsed_class,
                                              COLLAPSED_IDX[collapsed_class], player_id))
                break

    for split, rows in rows_by_split.items():
        out_path = os.path.join(OUT_DIR, f"{split}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "native_class", "collapsed_class", "collapsed_idx", "player_id"])
            w.writerows(rows)
        print(f"[write] {out_path}: {len(rows)} clips, {len({r[4] for r in rows})} players")

    return rows_by_split


def verify_no_leakage(rows_by_split):
    player_sets = {split: {r[4] for r in rows} for split, rows in rows_by_split.items()}
    overlaps = {
        "train&val": player_sets["train"] & player_sets["val"],
        "train&test": player_sets["train"] & player_sets["test"],
        "val&test": player_sets["val"] & player_sets["test"],
    }
    for pair, overlap in overlaps.items():
        assert not overlap, f"PLAYER LEAKAGE across {pair}: {overlap}"
    print("[verify] zero player overlap across all 3 splits -- confirmed, not assumed")

    filepaths = [r[0] for rows in rows_by_split.values() for r in rows]
    assert len(filepaths) == len(set(filepaths)), "duplicate filepath across splits"
    print(f"[verify] {len(filepaths)} total clips, each appears in exactly one split")


def print_class_balance(rows_by_split):
    print("\n[class balance] collapsed_class counts per split:")
    header = f"{'class':<10}" + "".join(f"{s:>8}" for s in ("train", "val", "test"))
    print(header)
    for c in COLLAPSED_CLASSES:
        counts = [sum(1 for r in rows_by_split[s] if r[2] == c) for s in ("train", "val", "test")]
        print(f"{c:<10}" + "".join(f"{n:>8}" for n in counts))


def main():
    clips = collect_clips()
    print(f"[setup] {len(clips)} total clips across {len(NATIVE_TO_COLLAPSED)} native classes")

    player_split = split_players(clips)
    for split, players in player_split.items():
        print(f"[split] {split}: {len(players)} players")

    rows_by_split = write_manifests(clips, player_split)
    verify_no_leakage(rows_by_split)
    print_class_balance(rows_by_split)


if __name__ == "__main__":
    main()
