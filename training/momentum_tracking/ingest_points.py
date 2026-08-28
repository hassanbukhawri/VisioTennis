"""Model 3 (momentum tracking) -- Step 1 of the real pipeline: point-level ingestion.

Scope (per user instruction, 2026-07-28): ingestion + EWMA label computation +
sanity check ONLY. No ΔWP, no point-importance-weighting (both deferred until
EWMA is validated), no Dataset/DataLoader, no training loop.

Source data: `data/raw/charting-m-points-{2020s,2010s,to-2009}.csv` and
`data/raw/charting-m-matches.csv`, pulled directly from
raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/ --
the same source repo Tennalysis-web's existing MCP stats ingestion
(`D:\\Tennalysis-web\\data_pipeline\\stages\\load_charted_stats.py`) already
uses, confirmed via schema inspection in DATA_ACQUISITION_CHECKLIST.md before
this script was written. Cross-referenced against Tennalysis-web's live
`charted_match_stats` table (Postgres) to identify which charted matches are
already reconciled to real ATP player identities -- this script does NOT
re-run reconciliation, it reuses what already exists there.

Known raw-data quirk, root-caused before writing the fix (not guessed): rows
in the raw points CSVs are not guaranteed to be in `Pt` order within a
match_id -- about 18% of matches have out-of-order rows in the file, which is
just a row-ordering artifact (sorting by Pt alone cleanly resolves the large
majority). A much smaller number of matches (11 across all 3 files) have
genuine duplicate rows for the same Pt with differing shot-notation detail but
identical score/server/winner -- these are deduplicated on the score-relevant
columns. Anything that still isn't a clean 1..N sequence after both fixes is
dropped, not guessed at, and reported explicitly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_RAW = HERE / "data" / "raw"
DATA_PROCESSED = HERE / "data" / "processed"

CORE_COLS = ["match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#", "Svr", "PtWinner"]

# Half-life for the EWMA recent-win-rate momentum label, in points. Provisional --
# per DATA_ACQUISITION_CHECKLIST.md, this is approach #1 of 3 candidates, chosen
# for this first pass specifically because it needs no separate model to be built
# first (unlike #2/#3) -- the K/half-life choice itself is not validated yet,
# that's exactly what the sanity-check plots in this pass are checking.
EWMA_HALFLIFE_POINTS = 10


def load_raw_points(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    files = sorted(raw_dir.glob("charting-m-points-*.csv"))
    if not files:
        raise FileNotFoundError(f"no charting-m-points-*.csv files found in {raw_dir}")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def dedupe_and_sort(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix the raw-file row-ordering quirk and the rarer genuine-duplicate-row
    quirk, then verify explicitly that Pt is a clean 1..N sequence per match_id
    -- rather than assuming either fix worked."""
    matches_total_raw = df["match_id"].nunique()

    # Which matches actually needed a sort fix at all (diagnostic, not the fix itself)?
    unsorted_flags = df.groupby("match_id")["Pt"].apply(lambda s: not s.is_monotonic_increasing)
    matches_needed_sort_fix = int(unsorted_flags.sum())

    # Genuine duplicate-row fix: drop rows that are duplicates on every score-relevant
    # column (this leaves distinct shot-notation duplicates -- e.g. two charters'
    # differing '1st'/'2nd'/Notes text for what is otherwise the identical point --
    # alone as long as the columns that matter for the momentum label agree).
    deduped = df.drop_duplicates(subset=CORE_COLS).sort_values(["match_id", "Pt"]).reset_index(drop=True)

    def is_clean(pts: np.ndarray) -> bool:
        return len(pts) > 0 and pts[0] == 1 and np.all(np.diff(pts) == 1)

    clean_flags = deduped.groupby("match_id")["Pt"].apply(lambda s: is_clean(s.to_numpy()))
    good_ids = clean_flags[clean_flags].index
    bad_ids = clean_flags[~clean_flags].index.tolist()

    clean_df = deduped[deduped["match_id"].isin(good_ids)].reset_index(drop=True)

    # Explicit post-fix verification (the thing the user asked to confirm, not assume):
    verify = clean_df.groupby("match_id")["Pt"].apply(
        lambda s: s.is_monotonic_increasing and s.iloc[0] == 1 and (s.diff().dropna() == 1).all()
    )
    assert verify.all(), f"post-fix verification failed for {(~verify).sum()} matches -- dedupe/sort logic has a bug"

    report = {
        "matches_total_raw": matches_total_raw,
        "matches_needed_sort_fix": matches_needed_sort_fix,
        "matches_with_duplicate_rows": int((df.groupby("match_id")["Pt"].apply(lambda s: s.duplicated().any())).sum()),
        "matches_unresolvable": len(bad_ids),
        "unresolvable_match_ids": bad_ids,
        "matches_clean_final": len(good_ids),
        "pt_strictly_increasing_verified": bool(verify.all()),
    }
    return clean_df, report


def compute_ewma_momentum(df: pd.DataFrame, halflife: float = EWMA_HALFLIFE_POINTS) -> pd.DataFrame:
    """Momentum label, from player 1's perspective (Set1/Gm1 slot in the raw MCP
    schema): EWMA of the +1/-1 point-outcome series, per match, in point order.
    Positive = recent points trending player 1's way; negative = player 2's way.
    This is candidate approach #1 from DATA_ACQUISITION_CHECKLIST.md ONLY --
    approaches #2 (ΔWP) and #3 (point-importance-weighted) are explicitly out of
    scope for this pass."""
    df = df.copy()
    df["outcome_p1"] = np.where(df["PtWinner"] == 1, 1.0, -1.0)
    df["momentum_ewma_p1"] = (
        df.groupby("match_id")["outcome_p1"]
        .transform(lambda s: s.ewm(halflife=halflife, adjust=False).mean())
    )
    return df


def link_to_ingested_matches(points_df: pd.DataFrame, matches_csv: Path, tennalysis_root: Path) -> pd.DataFrame:
    """Flag which charted matches are already reconciled+stat-loaded in
    Tennalysis-web (queried live, not assumed from the earlier feasibility
    check), and attach real player names for readability."""
    matches_meta = pd.read_csv(matches_csv, low_memory=False)[
        ["match_id", "Player 1", "Player 2", "Date", "Tournament", "Round", "Best of"]
    ].rename(columns={"match_id": "charted_match_id"})

    sys.path.insert(0, str(tennalysis_root / "data_pipeline"))
    sys.path.insert(0, str(tennalysis_root / "data_pipeline" / "stages"))
    from db import get_engine  # noqa: E402
    from sqlalchemy import text  # noqa: E402

    engine = get_engine()
    with engine.connect() as conn:
        ingested = pd.read_sql(text("SELECT DISTINCT charted_match_id FROM charted_match_stats"), conn)
    ingested_ids = set(ingested["charted_match_id"])

    points_df = points_df.merge(
        matches_meta, left_on="match_id", right_on="charted_match_id", how="left"
    )
    points_df["already_ingested"] = points_df["match_id"].isin(ingested_ids)
    return points_df


def run(tennalysis_root: Path = Path(r"D:\Tennalysis-web")) -> dict:
    print("[1/4] loading raw points files...")
    raw = load_raw_points()
    print(f"      {len(raw):,} rows, {raw['match_id'].nunique():,} distinct matches")

    print("[2/4] dedupe + sort fix, with explicit post-fix verification...")
    clean, report = dedupe_and_sort(raw)
    for k, v in report.items():
        if k != "unresolvable_match_ids":
            print(f"      {k}: {v}")
    if report["unresolvable_match_ids"]:
        print("      unresolvable match_ids:")
        for mid in report["unresolvable_match_ids"]:
            print(f"        - {mid}")

    print("[3/4] computing EWMA momentum label (approach #1 only)...")
    labeled = compute_ewma_momentum(clean)

    print("[4/4] linking to Tennalysis-web's already-ingested matches...")
    linked = link_to_ingested_matches(labeled, DATA_RAW / "charting-m-matches.csv", tennalysis_root)
    n_ingested_matches = linked.loc[linked["already_ingested"], "match_id"].nunique()
    report["matches_linked_to_already_ingested"] = int(n_ingested_matches)
    print(f"      {n_ingested_matches:,} / {report['matches_clean_final']:,} clean matches are already-ingested (reconciled) ATP matches")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "points_ewma_momentum.parquet"
    linked.to_parquet(out_path, index=False)
    print(f"\nwrote {len(linked):,} rows ({linked['match_id'].nunique():,} matches) to {out_path}")

    return report


if __name__ == "__main__":
    run()
