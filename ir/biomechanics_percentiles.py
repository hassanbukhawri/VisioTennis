"""
Population-percentile benchmarking for stroke-level biomechanics, per
ir/DESIGN.md's decision: reference ranges are percentiles within our own
extracted feature population, grouped by collapsed stroke class -- no
external ITF/USTA/literature norms sourced, same precedent as the
session-level Model 4 decision.

**Built only after the full-dataset extraction pass (ir/populate_biomechanics_full.py)
was run and checked -- percentile validity depends on that being a real,
verified distribution, not assumed.** Concretely, this module is built
against `ir/example_output/biomechanics_full_dataset.csv`'s `clean`-bucket
rows (1670/1980 clips, 84.3%) -- clips excluded from that population
(insufficient valid frames, degenerate camera angle, degenerate arm
extension -- see that script's docstring) are excluded here too, since
their raw feature values were judged unreliable by the same full-dataset
run this percentile table is built from.

**Deliberately excludes 3 of the 10 BiomechanicsInfo fields from
percentile ranking, not just from this file's oversight:**
`shoulder_rotation_deg`, `hip_rotation_deg`, `forearm_orientation_deg` are
raw angles measured against a fixed frame-horizontal axis. Checked, not
assumed: their population std at full-dataset scale is 82-171 degrees per
class (essentially spanning the full -180..180 range) -- because which way
a player happens to face the camera in a given clip is an artifact of
filming setup, not a real biomechanical quantity, these are NOT comparable
across clips and a percentile rank against them would not mean anything.
`shoulder_hip_separation_deg` does not have this problem (it's already an
undirected AXIS-to-AXIS separation, see ir/biomechanics.py) and IS ranked.
`weight_transfer_ratio` has a milder version of the same caveat (its sign
depends on which screen-direction is "forward" for that clip, also
filming-setup-dependent) -- included here because a real class-level
pattern was observed (serve's mean is clearly separated from the other
classes), but the sign-consistency assumption is NOT independently
verified and should be treated as a caveat, not a guarantee, in any
generated coaching text that uses it.
"""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DATASET_CSV = os.path.join(REPO_ROOT, "ir", "example_output", "biomechanics_full_dataset.csv")

RANKED_FEATURES = (
    "wrist_height_ratio",
    "shoulder_hip_separation_deg",
    "knee_flex_left_deg",
    "knee_flex_right_deg",
    "stance_width_ratio",
    "weight_transfer_ratio",  # see module docstring caveat on sign/facing-direction
    "follow_through_extension_ratio",
    "elbow_angle_deg",
)
EXCLUDED_FEATURES = ("shoulder_rotation_deg", "hip_rotation_deg", "forearm_orientation_deg")

# Below this many clean samples for a class, tail percentiles (below ~10th
# or above ~90th) rest on very few points and should be reported with a
# low-confidence caveat -- not a hard cutoff, just a signal surfaced to
# the caller. Chosen consistent with the same "n=165/144 is thin for
# tails" judgment already made for the session-level smash class in
# ir/DESIGN.md's Step 2 percentile discussion.
LOW_CONFIDENCE_SAMPLE_SIZE = 150


class BiomechanicsPercentileBenchmark:
    """Loads the clean population once and answers "what percentile is
    this value, for this stroke class, for this feature" -- analogous to
    explain_anomaly()'s z-score lookup, but for biomechanics, and built
    once real per-class distributions exist and have been sanity-checked
    (see ir/DESIGN.md), not before."""

    def __init__(self, csv_path: str = FULL_DATASET_CSV):
        df = pd.read_csv(csv_path)
        self.clean = df[df["bucket"] == "clean"].copy()
        self._sorted_values = {}
        for cls, group in self.clean.groupby("collapsed"):
            for feat in RANKED_FEATURES:
                self._sorted_values[(cls, feat)] = np.sort(group[feat].to_numpy())

    def class_sample_size(self, collapsed_class: str) -> int:
        return int((self.clean["collapsed"] == collapsed_class).sum())

    def percentile(self, value: float, collapsed_class: str, feature: str) -> dict:
        """Returns {percentile, n, low_confidence, tail}. Raises ValueError
        for a feature that was deliberately excluded (see module docstring)
        rather than silently computing a meaningless number."""
        if feature in EXCLUDED_FEATURES:
            raise ValueError(
                f"{feature!r} is deliberately excluded from percentile ranking -- it is a raw, "
                f"camera-facing-direction-dependent angle, not comparable across clips. See "
                f"ir/biomechanics_percentiles.py module docstring."
            )
        key = (collapsed_class, feature)
        if key not in self._sorted_values:
            raise ValueError(f"no reference population for class={collapsed_class!r}, feature={feature!r}")

        arr = self._sorted_values[key]
        n = len(arr)
        rank = float(np.searchsorted(arr, value, side="left"))
        pct = 100.0 * rank / n
        tail = pct <= 10.0 or pct >= 90.0
        return dict(
            percentile=round(pct, 1),
            n=n,
            low_confidence=(n < LOW_CONFIDENCE_SAMPLE_SIZE) and tail,
            tail=tail,
        )


if __name__ == "__main__":
    bench = BiomechanicsPercentileBenchmark()
    print("Clean per-class sample sizes used as the percentile reference population:")
    for cls in ("forehand", "backhand", "serve", "volley", "smash"):
        print(f"  {cls:10s} n={bench.class_sample_size(cls)}")

    print("\nDemo: percentile lookup for the 5 hand-inspected examples from "
          "ir/populate_biomechanics.py (values hardcoded from that script's printed output):")
    demo_examples = [
        ("forehand", "wrist_height_ratio", 1.636),
        ("backhand", "wrist_height_ratio", 0.935),
        ("serve", "wrist_height_ratio", 5.651),
        ("volley", "wrist_height_ratio", -0.366),
        ("smash", "wrist_height_ratio", 0.652),
        ("backhand", "shoulder_hip_separation_deg", 12.89),
        ("serve", "elbow_angle_deg", 155.92),
    ]
    for cls, feat, val in demo_examples:
        result = bench.percentile(val, cls, feat)
        flag = " [LOW-CONFIDENCE TAIL ESTIMATE]" if result["low_confidence"] else ""
        print(f"  {cls:10s} {feat:30s} value={val:8.3f} -> {result['percentile']:5.1f}th percentile "
              f"(n={result['n']}){flag}")

    print("\nDemo: attempting to rank an excluded feature raises, rather than returning a "
          "meaningless number:")
    try:
        bench.percentile(90.0, "serve", "shoulder_rotation_deg")
    except ValueError as e:
        print(f"  ValueError: {e}")
