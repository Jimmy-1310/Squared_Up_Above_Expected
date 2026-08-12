"""Feature construction for the expectation model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OPTIONAL_FEATURES, SWING_PATH_FEATURES, Variant

__all__ = ["build_features"]


def _batter_side(df: pd.DataFrame) -> np.ndarray:
    """+1 for a right-handed batter, -1 for a left-handed one."""
    return np.where(df["stand"].astype(str).str.upper().str[0] == "R", 1.0, -1.0)


def build_features(df: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    """Pitch-level features.

    Two conventions matter:

    * **Handedness mirroring.** ``plate_x`` and ``pfx_x`` are from the
      catcher's view, so the same raw value means "inside" to a righty and
      "outside" to a lefty. Multiplying by the side indicator gives location
      relative to the batter.
    * **Zone normalisation.** ``(plate_z - sz_bot) / (sz_top - sz_bot)`` puts
      every hitter's strike zone on a common 0-1 scale, which matters across a
      5'6" and a 6'7" batter.

    Spin axis is circular (0 and 359 degrees are adjacent) so it enters as
    sine/cosine rather than raw degrees.
    """
    X = pd.DataFrame(index=df.index)
    side = _batter_side(df)

    X["plate_speed"] = df["plate_speed"]
    X["plate_x_adj"] = df["plate_x"] * side          # + = inside to this batter
    X["plate_z"] = df["plate_z"]

    zone_height = (df["sz_top"] - df["sz_bot"]).replace(0, np.nan)
    X["plate_z_norm"] = (df["plate_z"] - df["sz_bot"]) / zone_height

    X["pfx_x_adj"] = df["pfx_x"] * side
    X["pfx_z"] = df["pfx_z"]
    X["balls"] = df["balls"]
    X["strikes"] = df["strikes"]
    X["platoon_same"] = (
        df["p_throws"].astype(str).str[0] == df["stand"].astype(str).str[0]
    ).astype(int)

    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")

    if "spin_axis" in df.columns:
        radians = np.deg2rad(pd.to_numeric(df["spin_axis"], errors="coerce"))
        X["spin_axis_sin"] = np.sin(radians)
        X["spin_axis_cos"] = np.cos(radians)

    if variant.include_bat_speed:
        X["bat_speed"] = df["bat_speed"]

    if variant.include_swing_path:
        for col in SWING_PATH_FEATURES:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce")

    X["pitch_type"] = df["pitch_type"].astype("category")
    return X
