# Court Detection (Model 1B, segmentation) — dataset notes

## Split leakage found and fixed

The raw Roboflow export (`data/VisioTennis/Dataset/Court_Detector/`) ships a
`train`/`valid`/`test` split, but inspection showed it is a **frame-level random
split**, not a split by source video. Filenames reveal the export was assembled
from (at least) 4 distinct source-video batches, identifiable by naming pattern:

| source tag | total images | labeled |
|---|---|---|
| `{n}_jpg...` (bare) | 232 | 210 |
| `frame_{n}_jpg...` | 107 | 107 |
| `vid1_{n}_jpg...` | 133 | 132 |
| `vid2_{n}_jpg...` | 73 | 73 |

The original split scattered frames from the *same* source video across
train/valid/test — in the `vid1_`/`vid2_`/`frame_` groups, val frames sat as
little as 1–13 frames away from a train frame in the same video. Perceptual
hashing (16×16 average hash, Hamming distance over 256 bits) confirmed this is
real near-duplicate leakage, not just numerically-close frame IDs: **56% of the
original 107 validation images (60/107) were near-identical (Hamming ≤10) to a
training frame**, rising to 100% within the `vid1_` group alone.

This explains why the original run reported mAP50 = mAP50-95 = 0.995 — suspicious
because both thresholds landing on an identical value is a ceiling effect, and
the concern was that ceiling might be leakage rather than a genuinely strong
model. **It wasn't**: re-validating the trained model against only the `{n}_...`
(bare-numbered) group — the one group that was NOT tightly leaked (median frame-ID
gap 36, median Hamming distance ~102, i.e. essentially uncorrelated with its
nearest train frame) — gave the identical 0.995 mask mAP50/mAP50-95. So the
near-perfect score reflects a genuinely easy single-class segmentation task, not
memorization. It was still worth fixing the split, since the reported val/test
numbers should hold up on their own without needing that side-channel argument.

## The fix: group-level split (`prepare_data.py`)

`prepare_data.py` now pools all images from the raw `train`+`valid`+`test`
folders, groups them by source-video tag, and assigns each group **wholly** to
one split — no source video is ever split across train/valid/test:

```
train: {n}_... + vid1_...   (365 images, 342 labeled)
valid: frame_...            (107 images, 107 labeled)
test:  vid2_...             ( 73 images,  73 labeled)
```

With only 4 groups to distribute across 3 splits, the ratios (67.0% / 19.6% /
13.4%) don't land exactly on the conventional 70/20/10, but they're close, and
`valid` happens to land at exactly 107 images — the same size as the original
(leaky) validation set, so results are directly comparable by size.

## Result on the fixed split

Retrained YOLOv8n-seg for 100 epochs, identical config to the original run
(imgsz=640, batch=16). Best checkpoint (epoch 24, selected automatically by
val fitness):

| | mAP50 | mAP50-95 |
|---|---|---|
| box | 0.995 | 0.969 |
| mask | 0.995 | **0.995** |

Mask mAP50-95 matches the clean-subset control (0.995) exactly. Box mAP50-95
came in a bit below the original leaky-split run's 0.995 (which was itself
likely inflated for the box metric specifically), landing at 0.969 — still a
very strong result, and not a meaningful drop given the small eval set (107
images, 107 instances). Runs `court_seg` (original, leaky split) and
`court_seg_grouped` (fixed split, current) are both kept under `runs/` for
reference; `court_seg_grouped/weights/best.pt` is the model to use going
forward.

## Known limitation for the FYP writeup

**All current source footage comes from a small number of physical
courts/camera setups** (at most 4 source videos, based on the naming groups
above — plausibly fewer, since the naming scheme doesn't guarantee 4 distinct
physical locations). The train/valid/test split fix above ensures the reported
mAP is not inflated by frame-level memorization *within this footage*, but it
does **not** establish generalization to courts, camera angles, lighting, or
broadcast styles outside this footage. Read the reported mAP as **strong
same-domain performance**, not as validated cross-venue generalization — this
should be stated explicitly as a limitation in the FYP report, and ideally
revisited if footage from additional, genuinely distinct courts/venues becomes
available.
