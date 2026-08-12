"""Validation suite.

Every test iterates over ``cfg.variants``. None of them names a variant
directly. An earlier version hard-coded the non-working variant into two
tests and concluded the metric was worthless; the tests here are structured
so that mistake cannot recur silently.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import Config

__all__ = [
    "split_half_reliability",
    "season_aggregates",
    "year_over_year",
    "nested_r2",
    "sample_size_sweep",
    "permutation_spread_test",
]


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #
def split_half_reliability(
    bip: pd.DataFrame,
    column: str,
    group: str = "batter",
    min_n: int = 40,
    n_iter: int = 25,
) -> Dict[str, float]:
    """Split-half correlation with the Spearman-Brown correction.

    Randomly halve each player's batted balls, correlate the halves, then
    correct for the halved sample size: ``r_full = 2r / (1 + r)``.
    """
    correlations: List[float] = []
    for seed in range(n_iter):
        half = np.random.default_rng(seed).integers(0, 2, len(bip))
        a = bip[half == 0].groupby(group)[column].agg(["mean", "size"])
        b = bip[half == 1].groupby(group)[column].agg(["mean", "size"])
        joined = a.join(b, lsuffix="_a", rsuffix="_b").dropna()
        joined = joined[(joined["size_a"] >= min_n) & (joined["size_b"] >= min_n)]
        if len(joined) >= 15:
            correlations.append(float(np.corrcoef(joined["mean_a"], joined["mean_b"])[0, 1]))

    if not correlations:
        return {"r": np.nan, "spearman_brown": np.nan, "n_iter": 0}
    r = float(np.mean(correlations))
    return {"r": r, "spearman_brown": 2 * r / (1 + r), "n_iter": len(correlations)}


def reliability_report(bip: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Reliability of the raw metrics and every SUAX variant, side by side."""
    rows = [
        ("raw squared-up rate (binary)", "squared_up"),
        ("raw squared-up pct (continuous)", "squared_up_pct"),
    ]
    rows += [(f"SUAX · {v.name}", f"resid_{v.name}") for v in cfg.variants]

    records = []
    for label, col in rows:
        stats = split_half_reliability(bip, col, n_iter=cfg.split_half_iters)
        records.append({"metric": label, **stats})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Year over year
# --------------------------------------------------------------------------- #
def season_aggregates(bip: pd.DataFrame, year: int, cfg: Config) -> pd.DataFrame:
    """Per-hitter aggregates within a single season."""
    season = bip[bip["game_year"] == year]
    spec = {
        "bbe": ("squared_up_pct", "size"),
        "su_rate": ("squared_up", "mean"),
        "su_pct": ("squared_up_pct", "mean"),
        "bat_speed": ("bat_speed", "mean"),
    }
    for variant in cfg.variants:
        spec[f"r_{variant.name}"] = (f"resid_{variant.name}", "mean")
    if "estimated_woba_using_speedangle" in season.columns:
        spec["xwobacon"] = ("estimated_woba_using_speedangle", "mean")

    out = season.groupby("batter").agg(**spec)
    return out[out["bbe"] >= cfg.min_bbe_season]


def year_over_year(bip: pd.DataFrame, cfg: Config) -> Optional[pd.DataFrame]:
    """Join the first and last available seasons on hitters present in both.

    Same-season validation is contaminated: xwOBAcon is derived from exit
    velocity, and so is squared-up percentage, so the raw metric partly
    predicts itself. Season 1 -> season 2 is the uncontaminated test.
    """
    if "game_year" not in bip.columns:
        return None
    years = sorted(bip["game_year"].dropna().unique().astype(int))
    if len(years) < 2:
        return None

    first, last = years[0], years[-1]
    joined = season_aggregates(bip, first, cfg).join(
        season_aggregates(bip, last, cfg), lsuffix="_1", rsuffix="_2", how="inner"
    )
    if joined.empty:
        return None
    joined.attrs["year_1"] = first
    joined.attrs["year_2"] = last
    return joined


def nested_r2(yoy: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Incremental R-squared predicting season-2 xwOBAcon from season-1 features.

    Reports two framings, because they disagree and the disagreement is the
    honest result:

    * **Nested** -- SUAX added on top of bat speed *and* raw squared-up rate.
      The increment is small, because a regression containing both can already
      undo the denominator artifact linearly on its own.
    * **Parsimonious** -- bat speed alone versus bat speed plus SUAX. This is
      the increment that matters when SUAX is used as a standalone number.
    """
    if yoy is None or yoy.empty or "xwobacon_2" not in yoy.columns:
        return pd.DataFrame()
    if len(yoy) < 10:
        return pd.DataFrame()

    target = yoy["xwobacon_2"].values

    def r2(cols: List[str]) -> float:
        X = yoy[cols].fillna(yoy[cols].mean()).values
        return float(LinearRegression().fit(X, target).score(X, target))

    records = []
    cols: List[str] = ["bat_speed_1"]
    prev = 0.0
    for label, add in [("bat speed", None), ("+ raw squared-up rate", "su_rate_1")]:
        if add:
            cols = cols + [add]
        value = r2(cols)
        records.append({"framing": "nested", "model": label,
                        "r2": value, "delta_r2": value - prev})
        prev = value

    for variant in cfg.variants:
        cols = cols + [f"r_{variant.name}_1"]
        value = r2(cols)
        records.append({"framing": "nested", "model": f"+ SUAX ({variant.name})",
                        "r2": value, "delta_r2": value - prev})
        prev = value

    base = r2(["bat_speed_1"])
    records.append({"framing": "parsimonious", "model": "bat speed alone",
                    "r2": base, "delta_r2": np.nan})
    for variant in cfg.variants:
        value = r2(["bat_speed_1", f"r_{variant.name}_1"])
        records.append({"framing": "parsimonious",
                        "model": f"bat speed + SUAX ({variant.name})",
                        "r2": value, "delta_r2": value - base})

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Sample-size behaviour
# --------------------------------------------------------------------------- #
def sample_size_sweep(bip: pd.DataFrame, yoy: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """How each metric's predictive power varies with sample size.

    Draw random subsamples of season 1, compute each metric, correlate against
    the hitter's full season-2 contact quality.

    Read the **sign** before the slope. A metric with a negative correlation is
    predicting backwards and is unusable at any sample size.
    """
    if yoy is None or yoy.empty or "xwobacon_2" not in yoy.columns:
        return pd.DataFrame()

    year_1 = yoy.attrs.get("year_1")
    if year_1 is None:
        return pd.DataFrame()
    season = bip[bip["game_year"] == year_1]
    truth = yoy["xwobacon_2"]

    series = {"raw squared-up rate": "squared_up"}
    series.update({f"SUAX ({v.name})": f"resid_{v.name}" for v in cfg.variants})

    grouped = {b: g for b, g in season[season["batter"].isin(yoy.index)].groupby("batter")}
    records = []

    for n in cfg.sweep_sizes:
        eligible = [g for g in grouped.values() if len(g) >= n]
        if len(eligible) < 20:
            records.append({"n": n, "n_hitters": len(eligible),
                            **{k: np.nan for k in series}})
            continue

        accumulated = {k: [] for k in series}
        for rep in range(cfg.sweep_reps):
            rng = np.random.default_rng(1000 * rep + n)
            values = {k: [] for k in series}
            keep = []
            for batter, group in grouped.items():
                if len(group) < n:
                    continue
                subset = group.iloc[rng.choice(len(group), n, replace=False)]
                for key, col in series.items():
                    values[key].append(subset[col].mean())
                keep.append(batter)
            if len(keep) < 20:
                continue
            actual = truth.loc[keep].values
            for key in series:
                accumulated[key].append(np.corrcoef(values[key], actual)[0, 1])

        records.append({
            "n": n,
            "n_hitters": len(eligible),
            **{k: (float(np.mean(v)) if v else np.nan) for k, v in accumulated.items()},
        })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Is there any true spread?
# --------------------------------------------------------------------------- #
def permutation_spread_test(
    values: pd.Series,
    group_ids: pd.Series,
    min_n: int,
    n_perm: int = 400,
    seed: int = 42,
) -> Dict[str, float]:
    """Test whether between-group spread exceeds pure sampling noise.

    Shuffle residuals across groups while holding each group's size fixed, then
    recompute the between-group variance. That builds the null distribution of
    "what spread would we see if no player had any skill".

    Always run this on hitters as a positive control. If hitters also come back
    non-significant the test simply lacks power and a null result on pitchers
    means nothing.
    """
    frame = pd.DataFrame({"v": values.values, "g": group_ids.values}).dropna()
    sizes = frame.groupby("g")["v"].size()
    frame = frame[frame["g"].isin(sizes[sizes >= min_n].index)]

    if frame.empty:
        return {"observed": np.nan, "null_mean": np.nan, "p_value": np.nan, "n_groups": 0}

    def spread(f: pd.DataFrame) -> float:
        m = f.groupby("g")["v"].agg(["mean", "size"])
        w = m["size"] / m["size"].sum()
        return float(np.average((m["mean"] - np.average(m["mean"], weights=w)) ** 2, weights=w))

    observed = spread(frame)
    rng = np.random.default_rng(seed)
    shuffled = frame["v"].values.copy()
    group_col = frame["g"].values
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(shuffled)
        null[i] = spread(pd.DataFrame({"v": shuffled, "g": group_col}))

    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "p_value": float((null >= observed).mean()),
        "n_groups": int(frame["g"].nunique()),
    }
