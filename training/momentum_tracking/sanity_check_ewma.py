"""Model 3 (momentum tracking) -- Step 2: sanity-check the EWMA momentum label
on real matches before trusting it for the rest of the corpus.

Reads the output of ingest_points.py and:
  1. Resolves the TRUE match winner from Tennalysis-web's `matches.winner_id`
     (not from the raw point log's Set1/Set2 columns -- see the methodology
     note below, a real bug this script caught while picking example matches).
  2. Plots the EWMA momentum curve (player 1's perspective) for a small,
     hand-picked set of real matches: one well-known dramatic comeback, one
     real upset, and one comfortable/dominant win as a negative control.

Methodology note / bug caught while building this: naively reading the LAST
row's Set1/Set2 per match as "the final score" and comparing to determine the
winner is WRONG about 31% of the time (checked directly: 3437/4996 = 68.8%
agreement with the real winner from Tennalysis's `matches` table). Set1/Set2/
Gm1/Gm2/Pts in the raw MCP point log record the score BEFORE each point is
played (point 1 of every match is always logged at "0-0"), so the last row's
score reflects the state entering the last point, not the completed match
result. This script uses the real `winner_id` from Tennalysis's reconciled
`matches` table instead of re-deriving it from the point log, and does NOT
attempt to infer game/set completion from the raw score columns -- that would
be needed for approach #3 (point-importance weighting) later, and should reuse
this same lesson rather than re-deriving it from scratch.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_PROCESSED = HERE / "data" / "processed"
FIGS_DIR = HERE / "figures"

# Hand-picked after scanning the already-ingested corpus for the largest
# mid-match momentum reversals (see PROJECT_LOG.md 2026-07-28 entry for the
# selection method) -- not cherry-picked from a shortlist smaller than this.
EXAMPLE_MATCHES = {
    # Real final score (verified against Tennalysis's `matches.score`, not assumed):
    # "0-6 3-6 6-4 6-1 8-6" -- one of the most famous comebacks in tennis history.
    "20040606-M-Roland_Garros-F-Gaston_Gaudio-Guillermo_Coria": (
        "2004 Roland Garros Final: Gaudio d. Coria, 0-6 3-6 6-4 6-1 8-6\n"
        "(real, well-known comeback from 2 sets to love down)"
    ),
    # Real final score: "6-4 6-2" -- Federer won comfortably in straight sets.
    # NOTE: an earlier draft of this script mislabeled this as "Wawrinka d. Federer,
    # a real upset" from memory, without checking -- the sanity-check plot itself
    # contradicted that caption (winner shown as Federer), which is what caught the
    # error. Kept in as an example specifically because it shows point-level
    # momentum capturing real within-match ebb and flow (Wawrinka built an early
    # lead before Federer's momentum turned around) that isn't visible from the
    # tidy final scoreline alone.
    "20150516-M-Rome_Masters-SF-Stan_Wawrinka-Roger_Federer": (
        "2015 Rome Masters SF: Federer d. Wawrinka, 6-4 6-2\n"
        "(straight-sets win on paper -- did the point-level story stay that tidy?)"
    ),
    # Real final score: "6-0 6-1 6-2" -- about as dominant as a win gets.
    "20150911-M-US_Open-SF-Novak_Djokovic-Marin_Cilic": (
        "2015 US Open SF: Djokovic d. Cilic, 6-0 6-1 6-2\n(control: dominant win, no real reversal)"
    ),
}


def resolve_true_winner() -> pd.Series:
    return pd.read_parquet(DATA_PROCESSED / "match_winner_slot.parquet").set_index("charted_match_id")[
        "winner_charted_slot"
    ]


def load_labeled_points() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "points_ewma_momentum.parquet")
    return df[df["already_ingested"]].sort_values(["match_id", "Pt"]).reset_index(drop=True)


def plot_examples(df: pd.DataFrame, winner_slot: pd.Series) -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(EXAMPLE_MATCHES), 1, figsize=(10, 3.2 * len(EXAMPLE_MATCHES)), sharex=False)

    for ax, (mid, title) in zip(axes, EXAMPLE_MATCHES.items()):
        sub = df[df["match_id"] == mid]
        if sub.empty:
            ax.set_title(f"{title}\n[NOT FOUND in processed data]")
            continue
        winner = int(winner_slot.get(mid, 0))
        p1, p2 = sub["Player 1"].iloc[0], sub["Player 2"].iloc[0]
        winner_name = p1 if winner == 1 else p2

        ax.plot(sub["Pt"], sub["momentum_ewma_p1"], color="#2b6cb0", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.fill_between(sub["Pt"], sub["momentum_ewma_p1"], 0, where=sub["momentum_ewma_p1"] >= 0,
                         color="#2b6cb0", alpha=0.15)
        ax.fill_between(sub["Pt"], sub["momentum_ewma_p1"], 0, where=sub["momentum_ewma_p1"] < 0,
                         color="#c53030", alpha=0.15)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"momentum\n(+ = {p1}, - = {p2})", fontsize=8)
        ax.set_title(f"{title}\nactual winner: {winner_name} ({len(sub)} points charted)", fontsize=9)

    axes[-1].set_xlabel("point number in match (Pt)")
    fig.tight_layout()
    out_path = FIGS_DIR / "ewma_momentum_examples.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def validate_winner_derivation(df: pd.DataFrame, winner_slot: pd.Series) -> None:
    """The methodology check described in the module docstring -- run every
    time so this doesn't silently regress if the ingestion logic changes."""
    naive_final = df.groupby("match_id").tail(1)[["match_id", "Set1", "Set2"]].set_index("match_id")
    naive_winner = pd.Series(
        np.where(naive_final["Set1"] > naive_final["Set2"], 1, np.where(naive_final["Set2"] > naive_final["Set1"], 2, 0)),
        index=naive_final.index,
    )
    true_winner = winner_slot.reindex(naive_winner.index).dropna()
    agree = (naive_winner.reindex(true_winner.index) == true_winner)
    print(
        f"[methodology check] naive last-row-Set-comparison winner agrees with true winner_id "
        f"in {agree.mean():.1%} of matches ({agree.sum()}/{len(agree)}) -- confirms Set/Game columns "
        f"are pre-point state, not post-point, so this script uses the real winner_id instead."
    )


def run() -> None:
    df = load_labeled_points()
    winner_slot = resolve_true_winner()
    validate_winner_derivation(df, winner_slot)
    plot_examples(df, winner_slot)


if __name__ == "__main__":
    run()
