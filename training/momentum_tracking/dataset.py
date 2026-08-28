"""
Model 3 (momentum tracking) -- feature engineering + PyTorch Dataset,
building on ingest_points.py's already-validated EWMA label
(points_ewma_momentum.parquet, ~4,996 real-ATP-linked matches, sanity-
checked against 3 known real matches including the negative control).

Feature set: adapted from model.py's provisional 13-dim spec against the
REAL confirmed schema (columns actually present: match_id, Pt, Set1/Set2,
Gm1/Gm2, Pts, Gm#, TbSet, Svr, 1st, 2nd, PtWinner -- no pre-computed
break/set/match-point flags, no rally-length, no outcome-type columns;
these all have to be derived, same as every other model in this project
that had to build features from raw fields rather than assume a
ready-made schema).

Deviations from the original 13-dim spec, and why:
  - Everything is in PLAYER 1's frame throughout (server, point_winner,
    game/set-score-diff, and the point/set/match-point flags below are all
    SIGNED: +1 favors player 1, -1 favors player 2, 0 neither) -- matches
    how momentum_ewma_p1 itself is defined (ingest_points.py:
    compute_ewma_momentum, strictly player-1-relative), not a symmetric
    "point of interest" abstraction the original spec's docstring implied
    but never pinned down. A signed feature is strictly more informative
    than the original's flat 0/1 and costs nothing extra to compute once
    the P1/P2 framing is fixed anyway.
  - point_outcome_ace / point_outcome_winner / point_outcome_unforced_error:
    derived from the last character of whichever of `1st`/`2nd` is
    non-null, EMPIRICALLY VALIDATED against real PtWinner before trusting
    it (not assumed from memory of the MCP notation spec): '#' correlates
    with the server winning 77.5% of the time (n=272,465) -- consistent
    with an unreturned/ace-type ending; '@' correlates at 48.4% (n=261,538,
    near coin-flip) -- consistent with an unforced error, whoever hit the
    last shot then loses the point roughly independent of who's serving;
    '*' at 72.9% (n=257,491) -- consistent with a clean winner. These 3
    codes cover the overwhelming majority (94%+) of terminal characters.
  - point_outcome_forced_error DROPPED, not guessed: no terminal character
    in the real data cleanly separated out as a distinct "forced error"
    marker distinguishable from the other codes without further unverified
    assumption -- rather than invent a mapping not confirmed against real
    outcomes, this sub-feature is omitted. If forced-error detail turns out
    to matter for a later revision, it needs a real MCP notation reference
    doc consulted directly, not a guess.
  - stroke_type_onehot: not included -- explicitly optional in the
    original spec, requires per-point video alignment with Model 2's
    stroke classifier, which does not exist for this charted-match corpus
    (a completely different data source than the stroke-classification
    video clips).
  - rally_length: no exact shot-count parser was built (the MCP shot
    grammar is more involved than the terminal-character check above and
    wasn't the goal here) -- used a rough, honestly-labeled PROXY instead:
    the character length of the combined `1st`+`2nd` notation string,
    normalized by the corpus's own 95th percentile length. Correlates with
    real rally length but is not an exact shot count.
  - is_break_point / is_set_point / is_match_point: derived from Pts
    (server-relative "0"/"15"/"30"/"40"/"AD" strings, or raw tiebreak
    integers when TbSet=True) + Gm1/Gm2 + Set1/Set2 + Best of, using the
    standard modern ATP rule (best-of-3 = first to 2 sets, best-of-5 =
    first to 3; sets won 6 games with a 2-game lead or a 7-6 tiebreak).
    Does NOT special-case deciding-set formats that don't use a
    match-tiebreak or use advantage sets (some majors/eras) -- a documented
    simplification, not exhaustively handled, given the time-box on this
    pass. Flagged here rather than silently assumed correct for every
    match in the corpus.

Sequence handling: match-level, variable length T (points per match range
widely -- 93 to 284+ in just the 3 sanity-check examples). Padded to the
batch's own max length with pack_padded_sequence-compatible lengths, not a
fixed global T -- avoids wasting compute on a fixed max forced by the
single longest match in the whole corpus.

--------------------------------------------------------------------------
LEAK-FREE REFRAME (2026-08-26), replacing the reconstruction-only version
above. The prior version's held-out MAE=0.00401 was a validated
reconstruction of the EWMA formula, not evidence of predictive power --
`point_winner` (the exact +1/-1 series `momentum_ewma_p1` is a plain EWMA
of) was a direct per-timestep input feature. See
`training/momentum_tracking/README.md` for the full prior scope statement;
this section documents the fix, not a repeat of that history.

FEATURE_NAMES now holds ONLY features describing match state ENTERING the
point in question -- confirmed real, not assumed: `Set1`/`Set2`/`Gm1`/`Gm2`/
`Pts` "record the score entering each point (point 1 of every match is
always logged at '0-0'), not the score after it" (DATA_ACQUISITION_CHECKLIST.md,
verified during ingestion, independent of this change) and `Svr` is the
server assignment, fixed by the alternating-service rule and therefore also
known before the point is played -- never derived from any point's outcome:
  server, game_score_diff, set_score_diff,
  is_break_point, is_set_point, is_match_point           (N_FEATURES = 6)

Everything derived from how the point itself was PLAYED --
`point_winner`, `rally_length_norm` (from the point's own shot notation),
`server_streak_norm` (previously computed inclusive of the current point's
own outcome -- see the old `build_match_features`, now fixed regardless of
its removal from the feature set), and the 3 `point_outcome_*` flags (from
the point's own terminal shot character) -- is still COMPUTED (kept in
`LEAKY_FEATURE_NAMES`, useful for diagnostics/plots) but excluded from the
model's input. `point_winner` in particular is the exact signal the EWMA
target is computed from -- keeping it as an input was the whole problem.

Target reframing: `MomentumDataset` now takes a `horizon` (points ahead,
default `DEFAULT_HORIZON`). At sequence position i (0-indexed, "context
available up to and not including point i+1"), the target is
`momentum_ewma_p1` at point `(i+1) + horizon - 1` -- i.e. `horizon=1` asks
"what will this point's own resulting momentum be, having seen only the
state entering it," `horizon=5` asks the same question 4 points further
out. The last `horizon-1` positions of each match have no valid target and
are truncated, not masked with a sentinel -- simpler, and avoids a second
mask semantic on top of the existing padding mask. Matches too short to
have at least 1 valid position after truncation are dropped and counted,
not silently skipped.
--------------------------------------------------------------------------
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(HERE, "data", "processed")
POINTS_PATH = os.path.join(DATA_PROCESSED, "points_ewma_momentum.parquet")

FEATURE_NAMES = [
    "server", "game_score_diff", "set_score_diff",
    "is_break_point", "is_set_point", "is_match_point",
]  # state entering the point -- leak-free, see module docstring
N_FEATURES = len(FEATURE_NAMES)

LEAKY_FEATURE_NAMES = [
    "point_winner", "rally_length_norm", "server_streak_norm",
    "point_outcome_ace", "point_outcome_winner", "point_outcome_unforced_error",
]  # still computed (diagnostics/plots), NEVER passed to the model -- each one
   # is derived from how the point itself was played, `point_winner` above
   # all: it's the exact +1/-1 series momentum_ewma_p1 is an EWMA of.

DEFAULT_HORIZON = 5  # points ahead the target is shifted; see module docstring

PTS_RANK = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def _terminal_outcome(row):
    s = row["2nd"] if isinstance(row["2nd"], str) and row["2nd"] else row["1st"]
    if not isinstance(s, str) or not s:
        return None
    return s[-1]


def _game_point_signed(pts, svr):
    """+1 if this point is a game/tiebreak point FOR player 1, -1 if FOR
    player 2 (i.e. player 1 would be broken/lose the TB), 0 otherwise.

    Deliberately does NOT use the `TbSet` column -- found by direct
    inspection (not assumed) that it's a SET-level flag ("did this set end
    in a tiebreak"), not a point-level "is this point currently inside the
    breaker" flag: it reads True for every point in a set from the very
    first 0-0 point onward, whenever that set eventually goes to 6-6.
    Trusting it here originally misparsed ordinary early-game scores like
    "15-0" as raw tiebreak point counts (15 points to 0), producing a
    game_point_signed=1 false positive on almost every point past 0-0 --
    caught by manually checking a real game's point sequence against the
    computed flag, not assumed correct from the aggregate rate alone.
    Fixed by detecting tiebreak-style scoring directly from `pts` itself
    (values outside the standard 0/15/30/40/AD vocabulary), which doesn't
    depend on correctly interpreting what TbSet means at all."""
    try:
        a_s, b_s = pts.split("-")
    except (ValueError, AttributeError):
        return 0
    if a_s in PTS_RANK and b_s in PTS_RANK:
        a, b = PTS_RANK[a_s], PTS_RANK[b_s]
        server_gp = a >= 3 and a > b
        returner_gp = b >= 3 and b > a
    else:
        try:
            a, b = int(a_s), int(b_s)
        except ValueError:
            return 0
        # TB win: first to >=7 with a 2-point lead. "Game point" = one point
        # from clinching for whichever side is ahead-or-tied at >=6 already.
        server_gp = a >= 6 and a - b >= 1
        returner_gp = b >= 6 and b - a >= 1
    if not server_gp and not returner_gp:
        return 0
    server_is_p1 = svr == 1
    if server_gp:
        return 1 if server_is_p1 else -1
    return -1 if server_is_p1 else 1


def _set_clinching(games_if_won, games_other):
    return (games_if_won >= 6 and games_if_won - games_other >= 2) or (games_if_won == 7 and games_other == 6)


def _set_point_signed(game_point_signed, gm1, gm2):
    """`gm1==6 and gm2==6` (both players' CONFIRMED, clean game counts,
    not the point-level TbSet flag -- see _game_point_signed's docstring
    for why that flag's semantics don't mean what its name suggests) is
    the standard-rule trigger for a set tiebreak: winning the breaker
    itself wins the set, so the tiebreak's own game-point IS the set
    point, no separate games-based clinching check needed."""
    if game_point_signed == 0:
        return 0
    if gm1 == 6 and gm2 == 6:
        return game_point_signed
    if game_point_signed == 1:
        return 1 if _set_clinching(gm1 + 1, gm2) else 0
    return -1 if _set_clinching(gm2 + 1, gm1) else 0


def _match_point_signed(set_point_signed, set1, set2, best_of):
    if set_point_signed == 0:
        return 0
    sets_needed = 2 if best_of == "3" else 3
    if set_point_signed == 1:
        return 1 if (set1 + 1) >= sets_needed else 0
    return -1 if (set2 + 1) >= sets_needed else 0


def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: one match's rows, already sorted by Pt. Returns df with the
    N_FEATURES feature columns added, plus the momentum_ewma_p1 target."""
    df = df.copy()
    df["server"] = (df["Svr"] == 1).astype(np.float32) * 2 - 1  # +1 P1 serves, -1 P2 serves
    df["point_winner"] = (df["PtWinner"] == 1).astype(np.float32) * 2 - 1
    df["game_score_diff"] = (df["Gm1"] - df["Gm2"]).astype(np.float32)
    df["set_score_diff"] = (df["Set1"] - df["Set2"]).astype(np.float32)

    game_point = [
        _game_point_signed(pts, svr) for pts, svr in zip(df["Pts"], df["Svr"])
    ]
    df["is_break_point"] = [
        gp if gp != 0 and ((gp == 1 and svr != 1) or (gp == -1 and svr == 1)) else 0
        for gp, svr in zip(game_point, df["Svr"])
    ]
    df["is_set_point"] = [
        _set_point_signed(gp, g1, g2)
        for gp, g1, g2 in zip(game_point, df["Gm1"], df["Gm2"])
    ]
    df["is_match_point"] = [
        _match_point_signed(sp, s1, s2, bo)
        for sp, s1, s2, bo in zip(df["is_set_point"], df["Set1"], df["Set2"], df["Best of"])
    ]

    notation_len = (df["1st"].fillna("").astype(str) + df["2nd"].fillna("").astype(str)).str.len()
    df["rally_length_norm"] = notation_len.astype(np.float32)  # normalized corpus-wide later

    # Streak ENTERING each point (i.e. accumulated through the point before
    # it, not including its own outcome) -- recorded BEFORE updating
    # `running` with this row's own point_winner, unlike the pre-2026-08-26
    # version which recorded it after (a leak, now moot since this column
    # is excluded from the model's input anyway -- see LEAKY_FEATURE_NAMES
    # -- but fixed here too so the diagnostic column itself is honest).
    streak = np.zeros(len(df), dtype=np.float32)
    running = 0
    for i, won in enumerate(df["point_winner"]):
        streak[i] = running
        running = running + 1 if won > 0 else (running - 1 if running > 0 else -1) if won < 0 else 0
    df["server_streak_norm"] = streak  # normalized corpus-wide later

    terms = df.apply(_terminal_outcome, axis=1)
    df["point_outcome_ace"] = (terms == "#").astype(np.float32)
    df["point_outcome_winner"] = (terms == "*").astype(np.float32)
    df["point_outcome_unforced_error"] = (terms == "@").astype(np.float32)

    return df


def load_and_engineer(min_points=20) -> pd.DataFrame:
    """Loads the already-ingested (real ATP-linked) subset, applies feature
    engineering per match, drops matches shorter than min_points (too short
    for a meaningful momentum sequence -- retirements/walkovers charted as
    a handful of points). Returns one row per point, feature columns +
    momentum_ewma_p1 target + match_id."""
    df = pd.read_parquet(POINTS_PATH)
    df = df[df["already_ingested"]].sort_values(["match_id", "Pt"]).reset_index(drop=True)

    # Real, confirmed single-row data gap found while running the full corpus
    # (not the smoke-test subsample, which happened not to include it):
    # match 20160123-M-Australian_Open-R32-Stan_Wawrinka-Lukas_Rosol, Pt=59,
    # has Gm2=NaN in the raw MCP source file -- 1 row out of 823,874. Checked
    # the neighboring rows (Pt=58 and Pt=60, same game, Gm#=9.0) before fixing:
    # both read Gm2=0.0, confirming forward-fill is not a guess here, it's the
    # same value the point immediately before and after both actually have.
    # Applied within each match (sorted by Pt) so a gap is never filled from a
    # different match's trailing value.
    n_before = df["Gm1"].isna().sum() + df["Gm2"].isna().sum()
    df[["Gm1", "Gm2"]] = df.groupby("match_id")[["Gm1", "Gm2"]].transform(lambda s: s.ffill().bfill())
    n_after = df["Gm1"].isna().sum() + df["Gm2"].isna().sum()
    if n_before:
        print(f"[fix] forward/back-filled {n_before} NaN Gm1/Gm2 value(s) within-match "
              f"({n_after} remain unfixed)")

    lengths = df.groupby("match_id").size()
    keep_ids = lengths[lengths >= min_points].index
    df = df[df["match_id"].isin(keep_ids)].reset_index(drop=True)

    featured = df.groupby("match_id", group_keys=False).apply(build_match_features)

    # Corpus-wide normalization (not per-match -- consistent scale across matches).
    rl_p95 = featured["rally_length_norm"].quantile(0.95)
    featured["rally_length_norm"] = (featured["rally_length_norm"] / max(rl_p95, 1)).clip(0, 2).astype(np.float32)
    streak_p95 = featured["server_streak_norm"].abs().quantile(0.95)
    featured["server_streak_norm"] = (featured["server_streak_norm"] / max(streak_p95, 1)).clip(-2, 2).astype(np.float32)

    return featured


def match_level_split(match_ids: list, seed=42, train_frac=0.8, val_frac=0.1):
    """Match-level split -- a match's points must never cross train/val/test,
    same lesson as every other split-leakage fix in this project (THETIS
    clean-split, TenniSet player-disjoint split). Shuffled by match_id, not
    by row, so no point-level leakage is even possible by construction."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(match_ids)))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), \
        "match-level split leaked -- a match_id appears in more than one split"
    return train_ids, val_ids, test_ids


class MomentumDataset(Dataset):
    """One item = one match's point sequence: (T', N_FEATURES) leak-free
    input (state entering points 1..T'), (T',) target -- momentum_ewma_p1
    shifted `horizon` points ahead (position i's target is the EWMA value
    at point i+horizon, 1-indexed), T' = that match's real point count minus
    (horizon-1) trailing points with no valid future target. No padding
    here -- padding/masking happens in the collate_fn at batch time, since
    T' varies match to match."""

    def __init__(self, featured_df: pd.DataFrame, match_ids: set, horizon: int = DEFAULT_HORIZON):
        assert horizon >= 1
        self.horizon = horizon
        self.sequences = []
        self.targets = []
        self.match_ids = []
        dropped_too_short = 0
        for mid, g in featured_df[featured_df["match_id"].isin(match_ids)].groupby("match_id"):
            g = g.sort_values("Pt")
            t = len(g)
            usable_t = t - (horizon - 1)
            if usable_t < 1:
                dropped_too_short += 1
                continue
            x_full = g[FEATURE_NAMES].to_numpy(dtype=np.float32)
            y_full = g["momentum_ewma_p1"].to_numpy(dtype=np.float32)
            x = x_full[:usable_t]
            y = y_full[horizon - 1: horizon - 1 + usable_t]
            self.sequences.append(x)
            self.targets.append(y)
            self.match_ids.append(mid)
        if dropped_too_short:
            print(f"[dataset] dropped {dropped_too_short} match(es) with fewer than "
                  f"{horizon} points (no valid horizon={horizon} target)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.from_numpy(self.targets[idx]),
            self.sequences[idx].shape[0],
        )


def collate_pad(batch):
    """Pads a batch of variable-length (T, F) sequences to the batch's own
    max T, returns lengths for pack_padded_sequence and a boolean mask for
    loss computation (padded timesteps must not contribute to the loss)."""
    xs, ys, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = int(lengths.max())
    b = len(xs)
    x_pad = torch.zeros(b, max_len, xs[0].shape[1], dtype=torch.float32)
    y_pad = torch.zeros(b, max_len, dtype=torch.float32)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    for i, (x, y, n) in enumerate(zip(xs, ys, lengths)):
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        mask[i, :n] = True
    return x_pad, y_pad, mask, lengths


if __name__ == "__main__":
    featured = load_and_engineer()
    print(f"[check] {featured['match_id'].nunique()} matches, {len(featured)} points after min-length filter")
    print(f"[check] feature means:\n{featured[FEATURE_NAMES].mean()}")
    train_ids, val_ids, test_ids = match_level_split(featured["match_id"].unique().tolist())
    print(f"[check] split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} matches")
