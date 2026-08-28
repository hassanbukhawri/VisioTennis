"""
Generates SessionRecords with DELIBERATELY DIFFERENT anomaly profiles, for
testing the RAG coaching layer's retrieval across varied anomaly types (see
rag/DESIGN.md Step 3, "validation across diverse sessions").

Why this exists: `ir/populate.py:populate_session_records()` takes the first
N anomaly-flagged rows out of one fixed synthetic dataset
(`make_synthetic_demo_data`, random_state=42). That dataset's "anomalous"
population is a single archetype (a generically bad/fatigued session where
every stat degrades together), so the first 3 flagged rows all end up with
overlapping top-3 deviated features (unforced_error_rate and
break_points_saved_pct dominate all three -- see session_record_0/1/2.json).
That's fine for comparing LLM backends on a fixed prompt, but useless for
testing whether retrieval generalizes across DIFFERENT anomaly types.

This script crafts 6 rows that each target a DIFFERENT feature theme (serve,
second-serve/ace, movement/fatigue, rally-style, clutch, general error
spike). The IsolationForest model, its fit, and explain_anomaly() are the
same real code as populate_session_records() -- only the crafted INPUT rows
are synthetic-by-design.

HONEST FINDING from building this (kept, not hidden, because it's a real
property of the fitted model worth noting for the FYP writeup): a pure
single-theme perturbation (2-3 features shifted, everything else left at
the exact population mean) is NOT reliably flagged anomalous by this
IsolationForest, even at extreme z (tested up to z=30) -- the fitted forest
was trained on an anomalous population that shifts ALL 12 features together,
so an isolated theme-only deviation often doesn't fall into the same
"isolated" leaves. Two of the six profiles below (serve_breakdown,
movement_fatigue) happen to trigger anomaly_flag=-1 on a pure theme shift;
the other four needed a mild additional blend toward the real anomalous
population's mean across ALL features (`blend_alpha`, e.g. 0.7) layered
under a much larger theme-specific override, so the row both (a) resembles
the trained "bad session" archetype closely enough to be isolated, and (b)
still has its theme feature(s) sitting furthest from the mean in z-score
terms, so explain_anomaly()'s top-3 ranking surfaces them first. This means
the 4 blended profiles' 3rd explanation slot sometimes picks up one
incidental spillover feature (e.g. a mild unforced_error_rate upticks) from
the blend rather than a second theme feature -- left visible in the
explanation text, not filtered out, since it's a genuine (if secondary)
z-score deviation on a real fitted model, not a fabricated value.
"""
import os
import sys
import importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

from ir.schema import ANOMALY_FEATURE_COLUMNS, ModelProvenance, SessionRecord  # noqa: E402


def _load_anomaly_train_module():
    path = os.path.join(REPO_ROOT, "training", "anomaly_detection", "train.py")
    spec = importlib.util.spec_from_file_location("anomaly_train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Physically-sane clip bounds per feature (rates/percentages can't go
# negative or, for percentages, above 100) -- applied after the z-score
# perturbation so a large synthetic shift can't produce an impossible value
# (e.g. a negative ace_rate) that would undermine an otherwise-real example.
FEATURE_BOUNDS = {
    "first_serve_pct": (0.0, 100.0),
    "first_serve_points_won_pct": (0.0, 100.0),
    "second_serve_points_won_pct": (0.0, 100.0),
    "ace_rate": (0.0, 1.0),
    "double_fault_rate": (0.0, 1.0),
    "break_points_saved_pct": (0.0, 100.0),
    "unforced_error_rate": (0.0, 1.0),
    "winners_rate": (0.0, 1.0),
    "avg_rally_length": (1.0, None),
    "court_coverage_speed_mps": (0.0, None),
    "avg_reaction_time_ms": (0.0, None),
    "total_distance_covered_m": (0.0, None),
}

# Each profile: theme_z = {feature: target_z_score} for the headline
# deviation(s); blend_alpha in [0, 1) optionally blends the row toward the
# real fitted anomalous population's mean across ALL features first (see
# module docstring for why some profiles need this to actually get flagged
# anomalous by the IsolationForest).
DIVERSE_PROFILES = {
    "serve_breakdown": dict(
        blend_alpha=0.0,
        theme_z={
            "first_serve_pct": -6.0,
            "first_serve_points_won_pct": -6.0,
            "double_fault_rate": 6.0,
        },
    ),
    "movement_fatigue": dict(
        blend_alpha=0.0,
        theme_z={
            "court_coverage_speed_mps": -6.0,
            "avg_reaction_time_ms": 6.0,
            "total_distance_covered_m": -6.0,
        },
    ),
    "second_serve_ace_decline": dict(
        blend_alpha=0.7,
        theme_z={
            "second_serve_points_won_pct": -6.0,
            "ace_rate": -6.0,
        },
    ),
    "rally_style_shift": dict(
        blend_alpha=0.7,
        theme_z={
            "avg_rally_length": -6.0,
            "winners_rate": 6.0,
        },
    ),
    "clutch_breakdown": dict(
        blend_alpha=0.7,
        theme_z={
            "break_points_saved_pct": -6.0,
        },
    ),
    "unforced_error_spike": dict(
        blend_alpha=0.7,
        theme_z={
            "unforced_error_rate": 6.0,
        },
    ),
}


def _clip(name, value):
    lo, hi = FEATURE_BOUNDS[name]
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def make_profile_row(mean, std, anom_mean, feature_columns, blend_alpha, theme_z):
    base = mean + blend_alpha * (anom_mean - mean)
    row = dict(zip(feature_columns, base))
    for name, z in theme_z.items():
        idx = feature_columns.index(name)
        row[name] = mean[idx] + z * std[idx]
    return {name: _clip(name, float(val)) for name, val in row.items()}


def populate_diverse_session_records(random_state=42):
    anomaly_train_mod = _load_anomaly_train_module()
    df = anomaly_train_mod.make_synthetic_demo_data(random_state=random_state)
    feature_columns = list(ANOMALY_FEATURE_COLUMNS)
    X = df[feature_columns].values.astype(float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    anom_mean = df[df["is_synthetic_anomaly"] == 1][feature_columns].values.astype(float).mean(axis=0)

    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=random_state, n_jobs=-1)
    model.fit(X)

    records = []
    for name, profile in DIVERSE_PROFILES.items():
        row_dict = make_profile_row(
            mean, std, anom_mean, feature_columns, profile["blend_alpha"], profile["theme_z"])
        row_vec = np.array([row_dict[c] for c in feature_columns]).reshape(1, -1)

        flag = int(model.predict(row_vec)[0])
        score = float(model.decision_function(row_vec)[0])
        explanation = anomaly_train_mod.explain_anomaly(
            pd.Series(row_dict), mean, std, feature_columns, top_k=3)

        records.append((name, SessionRecord(
            player_id=f"synthetic_diverse_{name}",
            features={c: float(row_dict[c]) for c in feature_columns},
            anomaly_flag=flag,
            anomaly_score=score,
            explanation=explanation,
            source_model=ModelProvenance(
                component="anomaly",
                checkpoint="IsolationForest (fit at population time, not persisted)",
                metric_name="contamination", metric_value=0.1,
                source_file=(
                    f"synthetic_diverse_profile '{name}' (crafted theme-targeted "
                    "perturbation, blend_alpha="
                    f"{profile['blend_alpha']}, NOT randomly sampled -- see "
                    "ir/populate_diverse_sessions.py DIVERSE_PROFILES)"
                ),
            ),
        )))
    return records


if __name__ == "__main__":
    out_dir = os.path.join(REPO_ROOT, "ir", "example_output")
    os.makedirs(out_dir, exist_ok=True)

    records = populate_diverse_session_records()
    n_flagged = 0
    for name, rec in records:
        out_path = os.path.join(out_dir, f"session_record_diverse_{name}.json")
        with open(out_path, "w") as f:
            f.write(rec.model_dump_json(indent=2))
        flagged_str = "ANOMALOUS" if rec.anomaly_flag == -1 else "NOT flagged (unexpected)"
        n_flagged += (rec.anomaly_flag == -1)
        print(f"[{name}] flag={rec.anomaly_flag} ({flagged_str}) score={rec.anomaly_score:.3f}")
        print(f"  explanation: {rec.explanation}")
        print(f"  -> {out_path}")

    print(f"\n[done] wrote {len(records)} diverse SessionRecords to {out_dir} "
          f"({n_flagged}/{len(records)} flagged anomalous)")
