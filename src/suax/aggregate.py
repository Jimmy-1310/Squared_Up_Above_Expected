"""Player-level aggregation with empirical-Bayes shrinkage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config

__all__ = ["ShrinkResult", "shrink", "build_hitter_board", "build_pitcher_board"]


@dataclass
class ShrinkResult:
    values: pd.Series
    k: float
    var_between: float
    var_sampling: float
    degenerate: bool


def shrink(
    mean_residual: pd.Series,
    n: pd.Series,
    within_var: float,
    label: str = "",
) -> ShrinkResult:
    """Shrink a per-player mean residual toward zero.

        theta_hat_i = r_bar_i * n_i / (n_i + k)
        k           = var_within / var_true
        var_true    = var_between - mean(var_within / n_i)

    ``k`` is interpretable as the sample size at which observed performance and
    league average are weighted equally.

    If ``var_true`` comes out non-positive, the observed spread is no larger
    than sampling noise would produce: there is no evidence of true talent
    differences. We flag it rather than silently returning a huge ``k``.
    """
    n = pd.Series(n).astype(float)
    if len(n) == 0 or not np.isfinite(n.sum()) or n.sum() <= 0:
        raise ValueError(
            f"shrink({label}): no qualifying players. Either the minimum "
            "batted-ball threshold is too high for this dataset, or the input "
            "is truncated. Check the qualification table before this call."
        )

    weights = n / n.sum()
    grand_mean = np.average(mean_residual, weights=weights)
    var_between = float(np.average((mean_residual - grand_mean) ** 2, weights=weights))
    var_sampling = float(np.mean(within_var / n))
    var_true = var_between - var_sampling

    degenerate = var_true <= 0
    if degenerate:
        var_true = 1e-12

    k = within_var / var_true
    return ShrinkResult(
        values=mean_residual * (n / (n + k)),
        k=float(k),
        var_between=var_between,
        var_sampling=var_sampling,
        degenerate=degenerate,
    )


def qualification_table(counts: pd.Series) -> pd.DataFrame:
    """How many players clear each candidate threshold."""
    thresholds = [25, 50, 100, 200, 300, 400]
    return pd.DataFrame(
        {"threshold": thresholds,
         "qualifying": [int((counts >= t).sum()) for t in thresholds]}
    )


def _aggregate(
    bip: pd.DataFrame,
    by: str,
    cfg: Config,
    min_bbe: int,
    names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    spec = {
        "bbe": ("squared_up_pct", "size"),
        "su_rate": ("squared_up", "mean"),
        "su_pct": ("squared_up_pct", "mean"),
        "bat_speed": ("bat_speed", "mean"),
        "plate_speed": ("plate_speed", "mean"),
    }
    for variant in cfg.variants:
        spec[f"r_{variant.name}"] = (f"resid_{variant.name}", "mean")
    if "estimated_woba_using_speedangle" in bip.columns:
        spec["xwobacon"] = ("estimated_woba_using_speedangle", "mean")
    if "swing_length" in bip.columns:
        spec["swing_length"] = ("swing_length", "mean")

    board = bip.groupby(by).agg(**spec)

    if (board["bbe"] >= min_bbe).sum() == 0:
        table = qualification_table(board["bbe"]).to_string(index=False)
        raise ValueError(
            f"no {by} reaches min_bbe={min_bbe} (largest is "
            f"{int(board['bbe'].max())}).\nQualification table:\n{table}\n"
            "A full season gives roughly 400 batted balls for a regular, so a "
            "much smaller maximum usually means a truncated input file."
        )

    board = board[board["bbe"] >= min_bbe].copy()
    if names:
        board.insert(0, "name", [names.get(int(i), str(int(i))) for i in board.index])
    return board


def build_hitter_board(
    bip: pd.DataFrame,
    cfg: Config,
    names: Optional[Dict[int, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Hitter leaderboard with shrunk SUAX for every variant."""
    board = _aggregate(bip, "batter", cfg, cfg.min_bbe, names)

    for variant in cfg.variants:
        within_var = float(bip[f"resid_{variant.name}"].var())
        res = shrink(board[f"r_{variant.name}"], board["bbe"], within_var, variant.name)
        board[f"suax_{variant.name}"] = res.values
        board[f"suax_{variant.name}_p100"] = res.values * 100
        if verbose:
            k_txt = "n/a" if res.degenerate else f"{res.k:.1f}"
            print(
                f"  {variant.name:12s} k={k_txt:>8s}  "
                f"var_between={res.var_between:.3e}  var_samp={res.var_sampling:.3e}"
                + ("  [DEGENERATE: spread indistinguishable from sampling noise; "
                   "all values shrink to ~0]" if res.degenerate else "")
            )
    return board


def build_pitcher_board(
    bip: pd.DataFrame,
    cfg: Config,
    names: Optional[Dict[int, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pitcher leaderboard: negative values indicate contact suppression.

    Read alongside ``validation.permutation_spread_test``. Pitchers do not face
    random hitters -- platoon usage, opponent quality and park all induce
    within-pitcher correlation that is not skill.
    """
    board = _aggregate(bip, "pitcher", cfg, cfg.min_bbe_pitcher, names)
    primary = cfg.primary
    within_var = float(bip[f"resid_{primary}"].var())
    res = shrink(board[f"r_{primary}"], board["bbe"], within_var, "pitchers")
    board["suabe"] = res.values
    board["suabe_p100"] = res.values * 100
    if verbose:
        k_txt = "n/a" if res.degenerate else f"{res.k:.1f}"
        print(f"  pitchers     k={k_txt:>8s}"
              + ("  [DEGENERATE: no detectable pitcher-level spread]"
                 if res.degenerate else ""))
    return board
