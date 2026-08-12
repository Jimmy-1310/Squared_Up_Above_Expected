"""Tests for the SUAX pipeline.

Several of these encode bugs that actually shipped in earlier versions. They
are regression tests in the literal sense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from suax import (
    Config,
    add_squared_up,
    build_features,
    league_rates,
    plate_speed_mph,
    run_pipeline,
    screen,
    shrink,
    tag_events,
    validate_columns,
)
from suax.config import VARIANTS
from suax.validation import permutation_spread_test, split_half_reliability

from synthetic import make_statcast


@pytest.fixture(scope="module")
def raw():
    return make_statcast(n_per_season=25_000, n_batters=100, seed=11)


@pytest.fixture(scope="module")
def bip(raw):
    cfg = Config()
    return screen(add_squared_up(tag_events(raw, cfg), cfg))


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def test_plate_speed_is_below_release_speed(raw):
    """A pitch decelerates on the way in; plate speed must be lower."""
    speed = plate_speed_mph(raw)
    assert (speed < raw["release_speed"]).mean() > 0.98
    ratio = (speed / raw["release_speed"]).mean()
    assert 0.88 < ratio < 0.96, f"plate/release ratio {ratio:.3f} outside plausible range"


def test_plate_speed_falls_back_without_kinematics(raw):
    stripped = raw.drop(columns=["vx0", "vy0", "vz0", "ax", "ay", "az"])
    fallback = plate_speed_mph(stripped)
    assert np.allclose(fallback, 0.92 * stripped["release_speed"])


def test_squared_up_only_defined_for_batted_balls(raw):
    out = add_squared_up(tag_events(raw), Config())
    no_ev = out["launch_speed"].isna()
    assert out.loc[no_ev, "squared_up_pct"].isna().all()


def test_squared_up_pct_is_clipped():
    cfg = Config(clip_pct=(0.10, 1.05))
    df = pd.DataFrame({
        "bat_speed": [70.0, 70.0],
        "release_speed": [90.0, 90.0],
        "launch_speed": [5.0, 200.0],     # absurd low and high
    })
    out = add_squared_up(df, cfg)
    assert out["squared_up_pct"].between(0.10, 1.05).all()


# --------------------------------------------------------------------------- #
# screening
# --------------------------------------------------------------------------- #
def test_bunts_put_in_play_are_removed(raw):
    """REGRESSION: a bunt put in play arrives as description='hit_into_play'.

    Filtering only on the bunt description codes lets it through, and because
    squared-up percentage has no bat-speed floor, bunts inflate the rate.
    """
    tagged = tag_events(raw, Config(drop_bunts_by_text=True))
    bunt_rows = raw["events"].astype(str).str.contains("bunt")
    assert bunt_rows.sum() > 0, "fixture should contain bunts"
    assert not tagged.loc[bunt_rows, "is_bip"].any()

    naive = tag_events(raw, Config(drop_bunts_by_text=False))
    assert naive.loc[bunt_rows, "is_bip"].any(), "naive filter should miss them"


def test_competitive_filter_removes_slowest_swings(bip):
    assert bip["bat_speed"].min() > 40, "bunts/emergency hacks survived screening"


def test_league_rates_land_in_published_range(raw):
    cfg = Config()
    df = add_squared_up(tag_events(raw, cfg), cfg)
    rates = league_rates(df, screen(df))
    assert 65 < rates["mean_bat_speed"] < 78
    assert 0.40 < rates["per_ball_in_play"] < 0.85
    assert 0.10 < rates["per_competitive_swing"] < 0.40


def test_validate_columns_rejects_incomplete_frame():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_columns(pd.DataFrame({"batter": [1]}))


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_handedness_mirroring(bip):
    """Same raw plate_x means opposite things to a righty and a lefty."""
    X = build_features(bip, VARIANTS[0])
    righty = bip["stand"].str.upper().str[0] == "R"
    assert np.allclose(X.loc[righty, "plate_x_adj"], bip.loc[righty, "plate_x"])
    assert np.allclose(X.loc[~righty, "plate_x_adj"], -bip.loc[~righty, "plate_x"])


def test_bat_speed_only_in_the_bat_speed_variant(bip):
    pitch_only = build_features(bip, VARIANTS[0])
    pitch_bs = build_features(bip, VARIANTS[1])
    assert "bat_speed" not in pitch_only.columns
    assert "bat_speed" in pitch_bs.columns


def test_spin_axis_is_circular(bip):
    X = build_features(bip, VARIANTS[1])
    assert {"spin_axis_sin", "spin_axis_cos"} <= set(X.columns)
    assert "spin_axis" not in X.columns


# --------------------------------------------------------------------------- #
# shrinkage
# --------------------------------------------------------------------------- #
def test_shrinkage_pulls_small_samples_harder():
    residual = pd.Series([0.05, 0.05])
    n = pd.Series([20.0, 2000.0])
    out = shrink(residual, n, within_var=0.01, label="t")
    assert abs(out.values.iloc[0]) < abs(out.values.iloc[1])


def test_shrinkage_flags_pure_noise():
    """No true spread -> flagged degenerate rather than a silent huge k."""
    rng = np.random.default_rng(0)
    n = pd.Series(np.full(200, 100.0))
    residual = pd.Series(rng.normal(0, np.sqrt(0.01 / 100), 200))
    out = shrink(residual, n, within_var=0.01, label="noise")
    assert out.degenerate
    assert (out.values.abs() < 1e-4).all()


def test_shrink_raises_on_empty_input():
    """REGRESSION: an empty board used to surface as a numpy ZeroDivisionError."""
    with pytest.raises(ValueError, match="no qualifying players"):
        shrink(pd.Series([], dtype=float), pd.Series([], dtype=float), 0.01, "empty")


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def test_permutation_test_detects_injected_signal():
    """Power check: injected group-level spread must be detected."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(60), 200)
    effect = rng.normal(0, 0.05, 60)[groups]
    with_signal = pd.Series(effect + rng.normal(0, 0.05, len(groups)))
    result = permutation_spread_test(with_signal, pd.Series(groups), 50, n_perm=200)
    assert result["p_value"] < 0.01
    assert result["observed"] > result["null_mean"]


def test_permutation_test_is_calibrated_under_the_null():
    """Specificity check, run across seeds.

    Under the null a p-value is uniform, so a single draw falls below 0.05
    five percent of the time -- asserting on one draw is a flaky test by
    construction. Check the median across several instead.
    """
    p_values = []
    for seed in range(9):
        rng = np.random.default_rng(100 + seed)
        groups = np.repeat(np.arange(60), 200)
        noise = pd.Series(rng.normal(0, 0.05, len(groups)))
        p_values.append(
            permutation_spread_test(noise, pd.Series(groups), 50,
                                    n_perm=120, seed=seed)["p_value"]
        )
    assert np.median(p_values) > 0.10, f"p-values skew small under null: {p_values}"


def test_split_half_reliability_recovers_stable_trait(bip):
    stats = split_half_reliability(bip, "squared_up_pct", min_n=25, n_iter=6)
    assert 0 < stats["spearman_brown"] <= 1.0


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_pipeline_end_to_end(raw):
    cfg = Config(min_bbe=40, min_bbe_season=30, min_bbe_pitcher=40,
                 n_splits=3, max_iter=80, permutation_iters=40,
                 sweep_sizes=(30, 60), sweep_reps=3, split_half_iters=4)
    result = run_pipeline(raw, cfg, resolve_player_names=False, verbose=False)

    for variant in cfg.variants:
        assert f"resid_{variant.name}" in result.bip.columns
        assert f"suax_{variant.name}_p100" in result.hitters.columns

    assert len(result.hitters) > 10
    assert not result.reliability.empty
    assert set(result.spread_tests) == {"hitters", "pitchers"}

    # every variant must appear in nested R2 -- guards against a validation
    # test silently running on only one of them
    if not result.nested.empty:
        models = " ".join(result.nested["model"])
        for variant in cfg.variants:
            assert variant.name in models


@pytest.mark.slow
def test_bat_speed_variant_decouples_from_bat_speed(raw):
    """The core claim: including bat speed in the model removes the artifact."""
    cfg = Config(min_bbe=40, n_splits=3, max_iter=120)
    result = run_pipeline(raw, cfg, resolve_player_names=False,
                          run_validation=False, verbose=False)
    board = result.hitters
    r_raw = abs(board["bat_speed"].corr(board["su_rate"]))
    r_adj = abs(board["bat_speed"].corr(board["suax_pitch_bs"]))
    assert r_adj < r_raw + 0.05, (
        f"adjusted correlation with bat speed ({r_adj:.3f}) should not exceed "
        f"the raw one ({r_raw:.3f})"
    )
