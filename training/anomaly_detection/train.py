"""
VisioTennis Model 4: player-session anomaly / fatigue detection (Isolation Forest).

CPU-only (sklearn) -- no GPU/ONNX concerns, per spec.

No historical multi-session performance dataset exists anywhere under
data/VisioTennis/Dataset/ as of this writing (checked: no session/anomaly-shaped
CSVs found). FEATURE_COLUMNS below is therefore a PLACEHOLDER schema, not
derived from a real file -- inspect and revise it against the actual schema
once a real per-session dataset is acquired, same rule as every other model.

Usage:
    # demo mode: runs the full pipeline against SYNTHETIC data (clearly not
    # real sessions) purely to prove the fit/score/explain pipeline works.
    python training/anomaly_detection/train.py --demo

    # real usage, once a real CSV exists with columns == FEATURE_COLUMNS
    # (+ a "player_id" and "session_date" column for bookkeeping, not used
    # as model features):
    python training/anomaly_detection/train.py --data path/to/sessions.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

# PLACEHOLDER feature schema -- one row per player-session. Every column here
# is a rate/percentage/speed aggregated over a single session, matching the
# spec's examples (first-serve %, court coverage speed, rally length).
FEATURE_COLUMNS = [
    "first_serve_pct",             # % of first serves in
    "first_serve_points_won_pct",  # % of points won on first serve
    "second_serve_points_won_pct", # % of points won on second serve
    "ace_rate",                    # aces per service point
    "double_fault_rate",           # double faults per service point
    "break_points_saved_pct",      # % of break points against saved
    "unforced_error_rate",         # unforced errors per point played
    "winners_rate",                # winners per point played
    "avg_rally_length",            # mean shots per rally
    "court_coverage_speed_mps",    # average movement speed, m/s (from tracking)
    "avg_reaction_time_ms",        # average time-to-first-move after opponent contact
    "total_distance_covered_m",    # total on-court distance covered in the session
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="CSV with columns == FEATURE_COLUMNS (+ player_id, session_date)")
    p.add_argument("--demo", action="store_true", help="run against synthetic data instead of --data")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--contamination", type=float, default=0.1, help="expected fraction of anomalous sessions")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--ckpt-dir", default=CKPT_DIR)
    p.add_argument("--top-k-explain", type=int, default=3)
    return p.parse_args()


def make_synthetic_demo_data(n_normal=180, n_anomalous=20, random_state=42):
    """SYNTHETIC data for demonstrating the pipeline only -- not a substitute for
    real sessions, not used to report any real-world result. Anomalous rows are
    generated as a distinct sub-population (worse serve/error/movement stats),
    mimicking a fatigued/off-day session, so explain_anomaly() has something
    real to point at."""
    rng = np.random.RandomState(random_state)

    normal = pd.DataFrame({
        "first_serve_pct": rng.normal(62, 5, n_normal),
        "first_serve_points_won_pct": rng.normal(70, 6, n_normal),
        "second_serve_points_won_pct": rng.normal(52, 6, n_normal),
        "ace_rate": rng.normal(0.08, 0.02, n_normal),
        "double_fault_rate": rng.normal(0.04, 0.015, n_normal),
        "break_points_saved_pct": rng.normal(60, 10, n_normal),
        "unforced_error_rate": rng.normal(0.15, 0.03, n_normal),
        "winners_rate": rng.normal(0.13, 0.03, n_normal),
        "avg_rally_length": rng.normal(4.5, 0.8, n_normal),
        "court_coverage_speed_mps": rng.normal(3.2, 0.3, n_normal),
        "avg_reaction_time_ms": rng.normal(210, 20, n_normal),
        "total_distance_covered_m": rng.normal(3200, 300, n_normal),
    })

    anomalous = pd.DataFrame({
        "first_serve_pct": rng.normal(48, 6, n_anomalous),
        "first_serve_points_won_pct": rng.normal(55, 7, n_anomalous),
        "second_serve_points_won_pct": rng.normal(40, 7, n_anomalous),
        "ace_rate": rng.normal(0.04, 0.02, n_anomalous),
        "double_fault_rate": rng.normal(0.10, 0.03, n_anomalous),
        "break_points_saved_pct": rng.normal(35, 12, n_anomalous),
        "unforced_error_rate": rng.normal(0.28, 0.05, n_anomalous),
        "winners_rate": rng.normal(0.08, 0.02, n_anomalous),
        "avg_rally_length": rng.normal(3.2, 0.7, n_anomalous),
        "court_coverage_speed_mps": rng.normal(2.3, 0.3, n_anomalous),
        "avg_reaction_time_ms": rng.normal(290, 30, n_anomalous),
        "total_distance_covered_m": rng.normal(2400, 350, n_anomalous),
    })

    df = pd.concat([normal, anomalous], ignore_index=True)
    df["player_id"] = [f"synthetic_p{i % 10}" for i in range(len(df))]
    df["session_date"] = pd.NaT
    df["is_synthetic_anomaly"] = [0] * n_normal + [1] * n_anomalous  # ground truth, demo only
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def explain_anomaly(row, mean, std, feature_columns, top_k=3):
    """Human-readable explanation: which features deviated most (in std-dev
    units) from the training distribution's mean. Simple and model-agnostic --
    not using IsolationForest's internal path-length structure, since that's
    harder to turn into a stable per-feature attribution; z-score deviation
    against the fitted training distribution is a reasonable first pass."""
    z = (row[feature_columns].values.astype(float) - mean) / std
    order = np.argsort(-np.abs(z))[:top_k]
    parts = []
    for i in order:
        col = feature_columns[i]
        direction = "above" if z[i] > 0 else "below"
        parts.append(f"{col}={row[col]:.2f} ({abs(z[i]):.1f} std {direction} average)")
    return "; ".join(parts)


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.demo:
        print("[data] --demo set: using SYNTHETIC session data. This demonstrates "
              "the pipeline only -- it is not a real result and must not be reported "
              "as one.", flush=True)
        df = make_synthetic_demo_data(random_state=args.random_state)
    elif args.data:
        df = pd.read_csv(args.data)
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{args.data} is missing expected columns: {sorted(missing)}")
        print(f"[data] loaded {len(df)} real sessions from {args.data}", flush=True)
    else:
        raise SystemExit("Pass --demo (synthetic pipeline check) or --data path/to/sessions.csv (real data).")

    X = df[FEATURE_COLUMNS].values.astype(float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0  # guard against a constant column

    model = IsolationForest(
        n_estimators=args.n_estimators, contamination=args.contamination,
        random_state=args.random_state, n_jobs=-1,
    )
    model.fit(X)

    flags = model.predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)  # higher = more normal, lower/negative = more anomalous

    df_out = df.copy()
    df_out["anomaly_flag"] = flags
    df_out["anomaly_score"] = scores

    n_flagged = int((flags == -1).sum())
    print(f"[result] {n_flagged}/{len(df)} sessions flagged anomalous "
          f"(contamination={args.contamination})", flush=True)

    print("\n[explain] example explanations for flagged sessions (first 5):", flush=True)
    flagged_idx = np.where(flags == -1)[0][:5]
    for i in flagged_idx:
        row = df_out.iloc[i]
        explanation = explain_anomaly(row, mean, std, FEATURE_COLUMNS, top_k=args.top_k_explain)
        pid = row.get("player_id", f"row{i}")
        print(f"  {pid}: score={row['anomaly_score']:.3f} -- {explanation}", flush=True)

    if args.demo:
        # sanity-check against the synthetic ground-truth labels -- demo only,
        # this comparison isn't meaningful/available on real unlabeled data.
        from sklearn.metrics import classification_report
        y_true = df["is_synthetic_anomaly"].values
        y_pred = (flags == -1).astype(int)
        print("\n[demo-only] recovery of injected synthetic anomalies "
              "(NOT a real-world metric, just confirms the pipeline works):", flush=True)
        print(classification_report(y_true, y_pred, target_names=["normal", "anomalous"], digits=3))

    out_path = os.path.join(args.ckpt_dir, "sessions_scored.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n[done] scored sessions written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
