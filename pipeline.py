"""End-to-end orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .aggregate import build_hitter_board, build_pitcher_board, qualification_table
from .config import Config
from .model import fit_all_variants
from .names import resolve_names
from .physics import add_squared_up
from .screening import league_rates, screen, tag_events, validate_columns
from .validation import (
    nested_r2,
    permutation_spread_test,
    reliability_report,
    sample_size_sweep,
    year_over_year,
)

__all__ = ["PipelineResult", "load_data", "run_pipeline"]


@dataclass
class PipelineResult:
    """Everything a run produces."""

    bip: pd.DataFrame
    hitters: pd.DataFrame
    pitchers: pd.DataFrame
    rates: Dict[str, float]
    model_scores: Dict[str, float]
    reliability: pd.DataFrame
    yoy: Optional[pd.DataFrame] = None
    nested: pd.DataFrame = field(default_factory=pd.DataFrame)
    sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    spread_tests: Dict[str, dict] = field(default_factory=dict)

    def write(self, outdir: str | Path) -> None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.hitters.round(4).to_csv(out / "suax_hitters.csv")
        self.pitchers.round(4).to_csv(out / "suax_pitchers.csv")
        self.reliability.round(4).to_csv(out / "reliability.csv", index=False)
        if self.yoy is not None:
            self.yoy.round(4).to_csv(out / "year_over_year.csv")
        if not self.nested.empty:
            self.nested.round(4).to_csv(out / "nested_r2.csv", index=False)
        if not self.sweep.empty:
            self.sweep.round(4).to_csv(out / "sample_size_sweep.csv", index=False)

        keep = ["batter", "pitcher", "game_year", "squared_up", "squared_up_pct",
                "bat_speed", "plate_speed"]
        keep += [c for c in self.bip.columns if c.startswith(("xpct_", "resid_"))]
        cols = [c for c in keep if c in self.bip.columns]
        self.bip[cols].round(4).to_csv(out / "suax_pitch_level.csv", index=False)
        print(f"wrote {len(list(out.glob('*.csv')))} files to {out}")


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a Statcast export. Parquet is strongly preferred over CSV.

    A two-season pull is roughly 1.4M rows -- about 1.5 GB as CSV and a few
    hundred MB as parquet. Use ``scripts/prep_statcast.py`` to build the
    parquet once, then load it in seconds.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def run_pipeline(
    data: str | Path | pd.DataFrame,
    cfg: Optional[Config] = None,
    resolve_player_names: bool = True,
    run_validation: bool = True,
    verbose: bool = True,
) -> PipelineResult:
    """Run the whole thing: physics, screening, model, boards, validation."""
    cfg = cfg or Config()
    raw = data if isinstance(data, pd.DataFrame) else load_data(data)
    validate_columns(raw)

    if verbose:
        print(f"loaded {len(raw):,} pitches")

    # --- physics and screening ------------------------------------------------
    df = add_squared_up(tag_events(raw, cfg), cfg)
    bip = screen(df)
    rates = league_rates(df, bip)

    if verbose:
        print(f"\n[league rates]  n_bip={rates['n_bip']:,}")
        print(f"  mean bat speed             {rates['mean_bat_speed']:.1f} mph   (MLB ~71.5)")
        print(f"  squared-up per comp. swing {rates['per_competitive_swing']:.1%}   (MLB ~25%)")
        print(f"  squared-up per ball in play {rates['per_ball_in_play']:.1%}   (expect 60-70%)")
        print("  NOTE: the published '33% of contacts' includes fouls, which have")
        print("        no launch_speed in public data. Do not compare against it.")
        print("\n[qualification]")
        print(qualification_table(bip.groupby('batter').size()).to_string(index=False))

    # --- names ----------------------------------------------------------------
    names: Dict[int, str] = {}
    if resolve_player_names:
        if verbose:
            print("\n[names]")
        ids = set(bip["batter"].dropna().astype(int)) | set(bip["pitcher"].dropna().astype(int))
        names = resolve_names(ids, verbose=verbose)

    # --- model ----------------------------------------------------------------
    if verbose:
        print("\n[expectation model]")
    model_scores = fit_all_variants(bip, cfg, verbose=verbose)

    # --- boards ---------------------------------------------------------------
    if verbose:
        print("\n[shrinkage]")
    hitters = build_hitter_board(bip, cfg, names, verbose=verbose)
    pitchers = build_pitcher_board(bip, cfg, names, verbose=verbose)

    result = PipelineResult(
        bip=bip, hitters=hitters, pitchers=pitchers,
        rates=rates, model_scores=model_scores,
        reliability=pd.DataFrame(),
    )

    if not run_validation:
        return result

    # --- validation -----------------------------------------------------------
    if verbose:
        print("\n[reliability]")
    result.reliability = reliability_report(bip, cfg)
    if verbose:
        print(result.reliability.round(3).to_string(index=False))

    result.yoy = year_over_year(bip, cfg)
    if result.yoy is None or result.yoy.empty:
        if verbose:
            n_years = bip["game_year"].nunique() if "game_year" in bip.columns else 0
            print(f"\n[year over year] skipped -- "
                  + ("need 2+ seasons of data" if n_years < 2 else
                     f"no hitter reaches min_bbe_season={cfg.min_bbe_season} in BOTH "
                     f"seasons; lower it or use more data"))
        result.yoy = None
    else:
        if verbose:
            print(f"\n[year over year] {len(result.yoy)} hitters in both seasons")
        result.nested = nested_r2(result.yoy, cfg)
        if verbose and not result.nested.empty:
            print(result.nested.round(4).to_string(index=False))
        result.sweep = sample_size_sweep(bip, result.yoy, cfg)
        if verbose and not result.sweep.empty:
            print("\n[sample-size sweep]")
            print(result.sweep.round(3).to_string(index=False))
            print("  Read the SIGN first: a negative correlation means the metric")
            print("  predicts backwards and is unusable at any sample size.")

    if verbose:
        print("\n[permutation spread tests]")
    primary_resid = bip[f"resid_{cfg.primary}"]
    result.spread_tests = {
        "hitters": permutation_spread_test(
            primary_resid, bip["batter"], cfg.min_bbe,
            cfg.permutation_iters, cfg.random_seed),
        "pitchers": permutation_spread_test(
            primary_resid, bip["pitcher"], cfg.min_bbe_pitcher,
            cfg.permutation_iters, cfg.random_seed),
    }
    if verbose:
        for group, stats in result.spread_tests.items():
            verdict = "real spread" if stats["p_value"] < 0.05 else "consistent with noise"
            print(f"  {group:9s} n={stats['n_groups']:4d}  observed={stats['observed']:.3e}  "
                  f"null={stats['null_mean']:.3e}  p={stats['p_value']:.4f}  -> {verdict}")
        print("  (hitters are the positive control: if they are not significant,")
        print("   the test lacks power and a pitcher null means nothing)")

    return result
