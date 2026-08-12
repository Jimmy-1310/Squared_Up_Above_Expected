"""Sample screening: swings, bunts, and Savant's competitive-swing rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BUNT_DESCRIPTIONS,
    IN_PLAY_DESCRIPTIONS,
    REQUIRED_COLUMNS,
    SWING_DESCRIPTIONS,
    Config,
)

__all__ = ["validate_columns", "tag_events", "screen", "league_rates"]


def validate_columns(df: pd.DataFrame) -> None:
    """Raise if the frame is missing anything the pipeline needs."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing required columns: {missing}\n"
            "Expected a raw Statcast pitch-level export. Bat tracking columns "
            "(bat_speed, swing_length) exist only from the second half of 2023."
        )


def tag_events(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add ``is_bunt``, ``is_swing``, ``is_bip``, ``is_competitive``.

    Two subtleties that cost real accuracy if skipped:

    1. A bunt *put in play* arrives as ``description == "hit_into_play"``. The
       bunt description codes never see it, so we also scan ``events``/``des``.
       Squared-up percentage has no bat-speed floor, so a bunt can register as
       "squared up" and inflate a hitter's rate.
    2. Savant's competitive-swing rule -- fastest 90% of a player's swings plus
       any 60+ mph swing producing 90+ mph exit velocity -- removes emergency
       hacks that are not representative of the hitter's intent.
    """
    cfg = cfg or Config()
    out = df.copy()
    desc = out["description"].astype(str)

    out["is_bunt"] = desc.isin(BUNT_DESCRIPTIONS)
    out["is_swing"] = desc.isin(SWING_DESCRIPTIONS) & ~out["is_bunt"]
    out["is_bip"] = desc.isin(IN_PLAY_DESCRIPTIONS) & ~out["is_bunt"]

    if cfg.drop_bunts_by_text:
        text = pd.Series("", index=out.index)
        for col in ("events", "des"):
            if col in out.columns:
                text = text + " " + out[col].astype(str).str.lower()
        out["is_bunt"] |= text.str.contains("bunt", na=False)
        out.loc[out["is_bunt"], ["is_swing", "is_bip"]] = False

    out["is_competitive"] = False
    swung = out["is_swing"] & out["bat_speed"].notna()
    if cfg.apply_competitive_filter:
        p10 = out.loc[swung].groupby("batter")["bat_speed"].quantile(0.10)
        threshold = out["batter"].map(p10)
        out.loc[swung, "is_competitive"] = (
            (out.loc[swung, "bat_speed"] >= threshold[swung])
            | (
                (out.loc[swung, "bat_speed"] >= 60)
                & (out.loc[swung, "launch_speed"].fillna(0) >= 90)
            )
        )
    else:
        out.loc[swung, "is_competitive"] = True

    return out


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling set: competitive swings put in play, with tracking."""
    return df[
        df["is_bip"] & df["squared_up_pct"].notna() & df["is_competitive"]
    ].copy()


def league_rates(df: pd.DataFrame, bip: pd.DataFrame) -> dict:
    """League-level sanity figures.

    ``per_competitive_swing`` is the one to check against MLB's published ~25%.

    Do NOT compare ``per_ball_in_play`` against the widely-quoted "33% of
    contacts": that denominator includes foul balls, and public Statcast has no
    ``launch_speed`` on fouls. Expect 60-70% in play.
    """
    n_comp = int(df["is_competitive"].sum())
    return {
        "n_bip": len(bip),
        "n_competitive_swings": n_comp,
        "mean_bat_speed": float(bip["bat_speed"].mean()),
        "per_ball_in_play": float(bip["squared_up"].mean()),
        "per_competitive_swing": float(bip["squared_up"].sum() / n_comp) if n_comp else np.nan,
        "mean_squared_up_pct": float(bip["squared_up_pct"].mean()),
    }
