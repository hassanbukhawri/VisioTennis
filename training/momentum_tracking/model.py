"""
VisioTennis Model 3: point-by-point momentum tracking (LSTM).

SCOPE: definitions only, per spec. No training loop, no dataset class, no
placeholder/fabricated data. Point-by-point match data (e.g. Match Charting
Project-style CSVs) is not present anywhere under data/VisioTennis/Dataset/ as
of this writing -- do not train against synthetic data standing in for it, and
do not wire this into a training script until real data is acquired (see
DATA_ACQUISITION_CHECKLIST.md in this folder).

SUPERSEDED, kept for history only -- the "Expected input" spec directly below
is the ORIGINAL 2026-07-17 provisional 14-dim scope, written before any real
data existed. `dataset.py`'s module docstring is authoritative for the actual
schema in use: real data wired up 2026-08-23 (input_dim 14 -> 12, real MCP
columns), then reframed leak-free 2026-08-26 (input_dim 12 -> 6, dropping
every point-outcome-derived feature -- `point_winner` above all, the exact
signal the training target is an EWMA of -- and shifting the target `horizon`
points ahead instead of reconstructing the same point's own value). See
`__init__`'s inline comment below and `training/momentum_tracking/README.md`
for why.
--------------------------------------------------------------------------
Expected input (ORIGINAL PROVISIONAL SPEC, superseded -- see above): a
sequence of per-point feature vectors within a single match,
shape (B, T, F) where:
  B = batch (matches, or sliding windows of points within a match)
  T = number of points in the sequence (variable length across matches --
      the training script, when written, will need to either pad+mask or
      window to a fixed T; not decided here since it depends on the real
      data's point-count distribution, which is unknown until acquired)
  F = per-point feature dimension, PROVISIONAL spec (subject to revision once
      real data is inspected -- follow the same "inspect before assuming"
      rule used for every other model in this project):

    0. server                  -- 1 if the point-of-interest player served, else 0
    1. point_winner             -- 1 if point-of-interest player won the point, else 0
    2. game_score_diff          -- (server games won - returner games won) in current set
    3. set_score_diff           -- (server sets won - returner sets won)
    4. is_break_point           -- 0/1
    5. is_set_point             -- 0/1
    6. is_match_point           -- 0/1
    7. rally_length             -- number of shots in the point, normalized
    8. point_outcome_ace        -- 0/1
    9. point_outcome_winner     -- 0/1
    10. point_outcome_unforced_error -- 0/1
    11. point_outcome_forced_error   -- 0/1
    12. server_streak           -- signed count of consecutive points won/lost by server
    13. stroke_type_onehot...   -- OPTIONAL, if Model 2 (stroke classification) is run
                                    on match video for the same match: one-hot over
                                    COLLAPSED_CLASSES (forehand/backhand/serve/volley/
                                    smash) for the shot that ended the point. Only
                                    include if per-point video alignment is available;
                                    the checklist below treats this as optional, not
                                    required, since the primary momentum signal is
                                    score-state, not stroke identity.

F above is a STARTING POINT for scoping the LSTM's input_dim, not a finalized
schema -- finalize it against whatever the acquired data actually contains
(same "inspect real files before writing the Dataset class" rule as every
other model here).

Output: per-timestep momentum score in [-1, 1] (positive = momentum toward the
point-of-interest player), via a Sigmoid-scaled head -- a regression target,
not a classification, since momentum is inherently continuous/smoothed rather
than discrete. The actual training target (how "momentum" is operationalized
as a label -- e.g. smoothed rolling win-rate over the next K points, a
break-probability model's output, or a human-annotated label) is NOT decided
here; it depends entirely on what the acquired dataset provides or what
derived target we choose to construct from raw point outcomes, which is a
modeling decision to make once real data is in hand, not before.
--------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TennisMomentumLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2):
        # input_dim default changed 13 -> 12 (2026-08-23, real data wired up) ->
        # 6 (2026-08-26, leak-free reframe): the 12-dim version included
        # `point_winner` and 5 other point-outcome-derived features (rally
        # length, ace/winner/UFE flags, server streak) that either directly
        # encode, or are only knowable after, the outcome of the point the
        # model was predicting momentum FOR -- confirmed to make the prior
        # MAE=0.00401 a validated reconstruction of the EWMA formula, not
        # evidence of predictive power (see dataset.py's module docstring
        # and training/momentum_tracking/README.md). The remaining 6 features
        # (server, game/set score diff, break/set/match-point flags) describe
        # match state ENTERING the point, never its outcome.
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),  # momentum score in [-1, 1]
        )

    def forward(self, x, lengths=None):
        """
        x: (B, T, input_dim) per-point feature sequence.
        lengths: optional (B,) actual sequence lengths. When given, uses
            pack_padded_sequence so the LSTM never sees/updates its hidden
            state on padded timesteps -- wired up now that train.py's
            variable-length, padded-batch data makes this a real
            requirement, not a hypothetical. When None (e.g. a single real
            match with no padding, at inference time), runs the LSTM
            directly over the full sequence.
        Returns: (B, T, 1) per-point momentum score in [-1, 1] (padded
            timesteps included in the output shape but meaningless -- the
            caller is responsible for masking, same convention as the
            training loss).
        """
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = self.lstm(x)  # (B, T, hidden_dim)
        return self.head(out)  # (B, T, 1)


if __name__ == "__main__":
    m = TennisMomentumLSTM()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TennisMomentumLSTM params: {n_params / 1e6:.3f}M")
    # shape check only -- input is random noise, NOT representative of real
    # feature distributions/ranges, since no real data exists to sample from.
    x = torch.randn(4, 20, 6)
    y = m(x)
    print("input:", x.shape, "-> output:", y.shape)
