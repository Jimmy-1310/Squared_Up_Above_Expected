"""Collision physics: pitch speed at the plate and squared-up percentage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    EVMAX_BAT_COEF,
    EVMAX_PITCH_COEF,
    FPS_TO_MPH,
    KINEMATIC_COLUMNS,
    PLATE_FRONT_Y,
    SQUARED_UP_THRESHOLD,
    STATCAST_Y0,
    Config,
)

__all__ = ["plate_speed_mph", "add_squared_up"]


def plate_speed_mph(df: pd.DataFrame) -> pd.Series:
    """Pitch speed at the front of home plate, in mph.

    The common shortcut is ``0.92 * release_speed``. When Statcast's kinematic
    constants are present we do better: position along the y-axis is

        y(t) = 50 + v_y0*t + 0.5*a_y*t^2

    Solve for y = 17/12 ft and evaluate the velocity magnitude there. That is
    the speed that actually enters the collision.

    Falls back to the 0.92 approximation when the kinematic columns are absent
    or produce a physically implausible value.
    """
    fallback = 0.92 * df["release_speed"]
    if not all(c in df.columns for c in KINEMATIC_COLUMNS):
        return fallback

    vy0 = pd.to_numeric(df["vy0"], errors="coerce")
    ay = pd.to_numeric(df["ay"], errors="coerce")

    disc = (vy0**2 - 2 * ay * (STATCAST_Y0 - PLATE_FRONT_Y)).clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-vy0 - np.sqrt(disc)) / ay

    vx = pd.to_numeric(df["vx0"], errors="coerce") + pd.to_numeric(df["ax"], errors="coerce") * t
    vy = vy0 + ay * t
    vz = pd.to_numeric(df["vz0"], errors="coerce") + pd.to_numeric(df["az"], errors="coerce") * t

    speed = np.sqrt(vx**2 + vy**2 + vz**2) * FPS_TO_MPH
    return speed.where(speed.between(50, 110), fallback)


def add_squared_up(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add ``plate_speed``, ``ev_max``, ``squared_up_pct`` and ``squared_up``.

    ``squared_up_pct`` is the continuous target and the one the model predicts.
    ``squared_up`` is the binary flag at 0.80 -- reported for comparison with
    published figures only. Modelling the binary costs roughly 0.13 of
    split-half reliability; see docs/METHODOLOGY.md section 4.3.
    """
    cfg = cfg or Config()
    out = df.copy()

    out["plate_speed"] = plate_speed_mph(out)
    out["ev_max"] = (
        EVMAX_BAT_COEF * out["bat_speed"] + EVMAX_PITCH_COEF * out["plate_speed"]
    )
    out["squared_up_pct"] = (out["launch_speed"] / out["ev_max"]).clip(*cfg.clip_pct)
    out["squared_up"] = (out["squared_up_pct"] >= SQUARED_UP_THRESHOLD).astype(float)

    missing = out["launch_speed"].isna() | out["bat_speed"].isna()
    out.loc[missing, ["squared_up_pct", "squared_up"]] = np.nan
    return out
