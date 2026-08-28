# Stroke Classification (Model 2) — dataset notes and results

## ⚠️ Split correction (2026-08-20) — read this before trusting any number below

The original train/val/test split documented in this file (**Results**
section, 84.5% accuracy / 0.834 macro-F1) had **100% player overlap across
all three splits** -- all 55 THETIS subjects appeared in train, val, AND
test simultaneously. This is the same class of leakage the court-detection
model was fixed for earlier in this project; it went uncaught here until
Path B's data-prep work (building a player-disjoint split for the video
model) surfaced it by direct comparison.

**A clean, player-disjoint re-split and retrain was run to check whether the
headline number holds.** It does not: on the corrected split (identical
39/8/8 player split used for Path B, zero player overlap, verified),
**test accuracy drops to 68.8% and macro-F1 to 0.647** -- see "Clean-split
retrain results" below for the full breakdown. **The clean numbers are the
current, trustworthy ones for the paper's headline result.** The original
84.5%/0.834 section further down is kept for historical/methodological
record (and because the confusion-matrix analysis of volley↔forehand
confusability in that section is still a real, useful finding, independent
of the split leakage), but should not be cited as the model's real
generalization performance.

## Source data

`data/VisioTennis/Dataset/StrokeClassification/processed/{train,val,test}/` —
`X_{split}.npy` is `(n_samples, 60, 99)` float32: 60-frame sequences of MediaPipe
Pose landmarks (33 keypoints × x/y/z = 99), already normalized, no NaNs. Filenames
in `labels_{split}.csv` (e.g. `p47_serflat_s3.avi`, `p20_serkick_s3.avi`) and the
55-player, 3-session, 12-class structure identify this as the public **THETIS**
tennis-stroke dataset. 1980 clips total, perfectly balanced at 165 clips/class
across the native 12 classes.

## Native 12 classes -> proposal's 5 categories

The FYP proposal targets 5 categories (forehand, backhand, serve, volley, smash)
with F1>82%, not THETIS's native 12. Every native class maps to exactly one
proposal category (decided with the user in chat) — **volley is its own
category** (forehand_volley + backhand_volley), not folded into forehand/backhand:

| proposal category | native classes | clip count |
|---|---|---|
| forehand | forehand_flat, forehand_openstands, forehand_slice | 495 |
| backhand | backhand, backhand2hands, backhand_slice | 495 |
| serve | flat_service, kick_service, slice_service | 495 |
| volley | forehand_volley, backhand_volley | 330 |
| smash | smash | 165 |

Training is done directly on these collapsed 5-class labels (`dataset.py:NATIVE_TO_COLLAPSED`),
not on the native 12 with post-hoc grouping — decided with the user since it
matches the proposal directly. This introduces class imbalance not present in
the native labels (smash is a real 3:1 minority vs. forehand/backhand/serve),
handled via inverse-frequency class weighting in the loss (`train.py`), not by
resampling or dropping data.

## Model

`StrokeGRU`: 2-layer bidirectional GRU (hidden=128) over the 60-frame sequence,
mean+max time-pooling, small FC head to 5 classes. 0.54M params — cheap enough
that the full 40-epoch training run completes in a few seconds on this GPU (no
need for the two-stream video fallback in the original spec; the pre-extracted
pose sequences are directly usable).

## Clean-split retrain results (trustworthy player-disjoint numbers -- base checkpoint for the deployed model, see "Serve investigation" below for what's actually live)

Player-disjoint split (`prepare_clean_pose_split.py`, reusing Path B's exact
`split_players` logic and seed -- 39/8/8 players, zero overlap, verified),
same model/hyperparameters/40-epoch budget as the original run, checkpoint
on best val macro-F1 (epoch 36/40, val_macro_f1=0.741). Evaluated on the
held-out clean **test** split (n=288, disjoint players from train/val,
`evaluate_clean_split.py`):

| | precision | recall | F1 | support |
|---|---|---|---|---|
| forehand | 0.827 | 0.597 | 0.694 | 72 |
| backhand | 0.764 | 0.764 | 0.764 | 72 |
| serve | 0.726 | 0.736 | 0.731 | 72 |
| volley | 0.617 | 0.771 | 0.685 | 48 |
| smash | 0.323 | 0.417 | 0.364 | 24 |
| **macro avg** | 0.651 | 0.657 | **0.647** | 288 |

**Test accuracy: 68.8%. Test macro-F1 = 0.647** -- both well below the
original leaky numbers (84.5% / 0.834) and below the proposal's F1>82%
target. Confusion matrix (rows=true, cols=pred):

| true \ pred | forehand | backhand | serve | volley | smash |
|---|---|---|---|---|---|
| **forehand** | 43 | 11 | 1 | 17 | 0 |
| **backhand** | 6 | 55 | 5 | 6 | 0 |
| **serve** | 0 | 1 | 53 | 0 | 18 |
| **volley** | 3 | 5 | 0 | 37 | 3 |
| **smash** | 0 | 0 | 14 | 0 | 10 |

**The drop is real, not noise.** Smash is now the weakest class by a wide
margin (F1 0.364, down from 0.809) -- half of true smashes are misclassified
as serve (14/24 in the confusion matrix), a confusion that barely existed in
the leaky split (6/25 in the original matrix, the other direction). Forehand
also drops sharply (0.828 -> 0.694), mostly lost to volley (17/72) the same
direction as the original run's genuine forehand/volley confusability
finding, just worse without the leaked player identity cues propping up
recall. Serve and backhand hold up best (F1 0.933->0.731 and 0.879->0.764
respectively) but both still drop meaningfully. **Interpretation: a
non-trivial share of the original 84.5%/0.834 was the model partially
recognizing individual players' body proportions/movement signatures across
splits, not purely the stroke technique** -- exactly the failure mode
player-disjoint splitting exists to catch. This is the number that should be
cited going forward; the FYP proposal's F1>82% target is not currently met
on a clean split.

Checkpoints: `checkpoints/clean_split/{stroke_gru_clean_best.pt,
stroke_gru_clean_last.pt}`. Confusion matrix also saved as
`checkpoints/clean_split/test_confusion_matrix.npy`.

---

## Original (leaky-split) results -- historical record, superseded above

Checkpoint on best **val macro-F1** (epoch 39/40, val_macro_f1=0.828).
Evaluated on the held-out **test** split (n=297, never used for training or
checkpoint selection):

| | precision | recall | F1 | support |
|---|---|---|---|---|
| forehand | 0.857 | 0.800 | 0.828 | 75 |
| backhand | 0.925 | 0.838 | 0.879 | 74 |
| serve | 0.909 | 0.959 | 0.933 | 73 |
| volley | 0.656 | 0.800 | 0.721 | 50 |
| smash | 0.864 | 0.760 | 0.809 | 25 |
| **macro avg** | 0.842 | 0.831 | **0.834** | 297 |

Test accuracy: 84.5%. **Test macro-F1 = 0.834, clearing the proposal's F1>82%
target.**

### Confusion matrix (test set, n=297)

Raw counts (rows = true class, columns = predicted class):

| true \ pred | forehand | backhand | serve | volley | smash |
|---|---|---|---|---|---|
| **forehand** | 60 | 2 | 0 | 13 | 0 |
| **backhand** | 3 | 62 | 1 | 8 | 0 |
| **serve** | 0 | 0 | 70 | 0 | 3 |
| **volley** | 7 | 3 | 0 | 40 | 0 |
| **smash** | 0 | 0 | 6 | 0 | 19 |

Normalized by true class (each row sums to 1.0):

| true \ pred | forehand | backhand | serve | volley | smash |
|---|---|---|---|---|---|
| **forehand** | 0.800 | 0.027 | 0.000 | 0.173 | 0.000 |
| **backhand** | 0.041 | 0.838 | 0.014 | 0.108 | 0.000 |
| **serve** | 0.000 | 0.000 | 0.959 | 0.000 | 0.041 |
| **volley** | 0.140 | 0.060 | 0.000 | 0.800 | 0.000 |
| **smash** | 0.000 | 0.000 | 0.240 | 0.000 | 0.760 |

Raw matrix also saved as `checkpoints/test_confusion_matrix.npy` (5x5 int array,
row/col order = `COLLAPSED_CLASSES` = forehand, backhand, serve, volley, smash).

### Is volley's lower F1 a data-volume effect or genuine confusability?

Volley support: **231 train / 50 test**. That's less than forehand/backhand/serve
(346-347 train / 73-75 test) but *more* than smash (115 train / 25 test) — and
smash still scores a higher F1 (0.809) than volley (0.721) despite having half
volley's training data. If the gap were mainly a data-volume effect, smash
should be the weaker class, not volley. **It isn't, so this points to genuine
feature confusability, not data volume.**

The confusion matrix confirms it directly: volley's errors are concentrated in
one direction — **7/50 (14.0%) of true volleys are predicted forehand**, another
3/50 (6.0%) predicted backhand, and volley is never confused with serve or
smash. The confusion is bidirectional with forehand specifically: 13/75 (17.3%)
of true forehands are predicted volley too — the single largest off-diagonal
cell in the whole matrix. Backhand↔volley confusion is smaller (8/74 = 10.8%
backhand→volley, 3/50 = 6.0% volley→backhand). Serve and smash are essentially
clean against everything except each other (serve↔smash, both overhead-motion
strokes: 3/73 serve→smash, 6/25 smash→serve) — the model isn't confusing
categories generally, it specifically struggles to separate volley from
forehand in pose-sequence space, most likely because a compact/open-stance
forehand and a forehand volley look kinematically similar from body pose alone
(no racket/ball trajectory signal in this feature set to disambiguate swing
length or contact point). Worth noting in the writeup: this is a feature-set
limitation, not something more training data would obviously fix.

`stroke_gru_best.pt` / `stroke_gru.onnx` (verified against the PyTorch model for
batch size 1 and 8, max abs diff ~1.9e-6) are the historical leaky-split
checkpoints -- superseded, not deployed. See "Serve investigation" below for
the actual deployed checkpoint and why.

## Serve investigation (2026-08-20 -> 2026-08-22): five interventions attempted, serve unresolved -- closed out as a documented limitation

**Starting point.** The clean-split retrain above (68.8% accuracy / 0.647
macro-F1 on THETIS's own held-out test split) already looked mediocre by
THETIS-internal standards. Against real, non-THETIS footage it is much
worse: a hand-labeled 14-instance real-footage validation set (3 serve, 10
backhand, 1 forehand -- no volley/smash coverage exists anywhere in this
project) built from independently-sourced amateur/practice clips shows the
deployed classifier scoring **5/14 (35.7%) overall, 0/3 on serve**, before
any of the interventions below. **0/3 real-footage serves correct is
consistent across every checkpoint tried in this investigation, with zero
exceptions** -- see the per-checkpoint table below.

Six interventions were tried, in this order, each validated against the
same fixed 14-instance set (never touched by any training data used in
any intervention) so results are directly comparable:

**1. Player-disjoint (clean) split retrain.** Fixed a real leaky-split bug
(100% player overlap between the original train/val/test) -- necessary
groundwork, not itself aimed at serve. Confirmed the model's THETIS-internal
number was inflated by leakage (84.5%->68.8%) but did not change its
real-footage behavior, since the leaky and clean checkpoints were never
directly real-footage-compared (the clean-split model became the baseline
all later work compares against).

**2. Post-hoc temperature-scaling calibration.** Fit per-checkpoint
(confirmed twice that a fitted T does NOT transfer between checkpoints --
each of the 3 checkpoints deployed this investigation needed its own
refit: T=1.4474 leaky-split, T=2.0046 clean-split, T=2.4741 this
checkpoint). Genuinely improves how honestly confidence is reported (ECE
reduced on every checkpoint tried) but is argmax-preserving by
construction -- it cannot and does not change which class is predicted, so
it was never going to fix a wrong-class prediction on its own. Included
here because it's a real, permanent part of what's deployed, not because
it targeted serve.

**3. Data augmentation targeting the diagnosed domain-shift gap**
(zoom-out/letterbox simulating camera distance, color jitter, motion blur,
noise, a background-clutter proxy) applied to THETIS's own training clips,
doubling the effective train set. **The only intervention that measurably
improved real-footage accuracy: 5/14 -> 8/14 (57.1%), entirely on backhand
(4/10 -> 7/10).** Serve: 0/3, unchanged. `clip5_serve_setup.mp4` (true
serve) was called backhand both before and after, with RAISED raw
confidence after augmentation (72.3% -> 100.0% raw, though calibration
brings the deployed number back down).

**4. Real (non-THETIS) broadcast footage from TenniSet** (5 annotated
professional matches, github.com/HaydenFaulkner/Tennis) -- extracted 3084
labeled clips (serve/forehand/backhand only; volley=62 and smash=17
confirmed too thin project-wide to use), a genuinely player-disjoint
split verified programmatically before extraction. **A major, unplanned
finding surfaced here: MediaPipe pose extraction itself failed on 75.3%
of the raw broadcast clips** (1974/2620) -- broadcast camera framing is
often too wide/distant for reliable pose tracking, independent evidence
for the same distance/framing hypothesis this whole investigation traces
back to. Surviving real volume was still substantial (302 serve, 261
backhand, 250 forehand clips after augmentation).

**5. Fine-tuning from the deployed clean-split checkpoint on THETIS(+aug) +
TenniSet(+aug) combined (3358 clips).** This was intended as the
definitive test. **Result: 5/14 (35.7%), tied with the pre-augmentation
baseline and WORSE than intervention #3 alone -- serve still 0/3,
backhand regressed (7/10 -> 4/10).** The internal validation metric
(val macro-F1) climbed to its highest value across all three checkpoints
built this investigation (0.741 -> 0.785 -> 0.798) while real-footage
accuracy was non-monotonic (35.7% -> 57.1% -> 35.7%) -- **the checkpoint
with the best internal validation score produced the worst real-world
result measured in this investigation.** Plausible, unconfirmed
explanation: TenniSet is professional broadcast footage (different camera
work, compression, framing) which may not actually be closer to this
app's real target domain (amateur/consumer-camera footage) than THETIS's
indoor footage is -- "real, non-THETIS video" was the tested hypothesis;
"video representative of this app's actual real-world input" was the
unproven assumption underneath it.

**6. Simple-averaging ensemble (clean-split + augmented checkpoints).**
Averaged both models' raw softmax probabilities per instance, argmaxed the
average. **Result: 8/14 (57.1%), exactly tied with the augmented
checkpoint alone -- identical correct/wrong pattern on 13 of 14 instances,
serve still 0/3.** The augmented checkpoint's sharper/more confident
predictions dominate the average whenever the two models disagree, so the
ensemble behaves as the augmented model alone in every case that affects
correctness. No measured benefit against a real added cost (two inference
passes instead of one) -- not deployed.

| checkpoint | internal val macro-F1 | real 14-instance accuracy | serve (real) |
|---|---|---|---|
| clean-split baseline | 0.741 | 5/14 (35.7%) | 0/3 |
| + THETIS augmentation (DEPLOYED, 2026-08-22) | 0.785 | **8/14 (57.1%)** | 0/3 |
| + TenniSet broadcast fine-tune | 0.798 (best) | 5/14 (35.7%) | 0/3 |
| ensemble (clean-split + augmented) | n/a | 8/14 (57.1%, tied) | 0/3 |

**Deployed model: intervention #3 (THETIS augmentation), as of 2026-08-22**
-- the only intervention that measurably helped anything, on the honest
understanding that it helps backhand/forehand and does not touch serve.
`checkpoints/augmented/stroke_gru_augmented.onnx`, calibrated with
T=2.4741 (`checkpoints/temperature.json`). The ensemble (#6) matched but
did not beat it, so the single augmented checkpoint remains deployed --
no added inference cost for no added accuracy.

**Conclusion, for the paper's limitations/results section:** across six
independently-motivated interventions -- a leakage fix, confidence
calibration, targeted synthetic augmentation, real broadcast training
data, a combination of the last two, and a two-model ensemble -- serve
classification accuracy on real, non-THETIS footage remained at 0/3 in
every single case. This rules out (not just fails to fix) several
plausible causes: it is not solely a train/test leakage artifact
(survives the clean-split fix), not solely a confidence-calibration
artifact (the model is confidently wrong, not just miscalibrated), not
fixed by simulated camera-distance/lighting/motion variation on the
existing training distribution, not fixed by adding real professional
broadcast serve footage at meaningful volume (302 clips), and not fixed
by combining models that make different errors (the ensemble adds no new
correct answers on serve). The most likely remaining explanation is a
genuine
distribution gap between every serve-labeled training source available
to this project (THETIS's controlled indoor serves, TenniSet's
professional broadcast serves) and the amateur/consumer-camera serve
footage this app actually needs to classify at deployment -- a gap this
project does not currently have the data to close, since no serve
footage matching the deployment distribution exists in sufficient volume
anywhere available to this project. **This is closed out as a documented,
accepted limitation.** No further serve-fixing attempts are planned
unless new real, deployment-representative serve footage becomes
available.
