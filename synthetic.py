"""Generate synthetic Statcast-shaped data with known ground truth.

Hitter barrel skill is injected as a fixed per-batter offset that persists
across seasons, so the year-over-year and reliability tests have something
real to recover. Bunts are injected deliberately so the screening tests can
confirm they are removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FPS_TO_MPH = 0.681818


def make_statcast(
    n_per_season: int = 40_000,
    seasons: tuple = (2024, 2025),
    n_batters: int = 120,
    n_pitchers: int = 90,
    n_bunts: int = 150,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.045, n_batters)       # persistent barrel-accuracy talent
    bat_talent = rng.normal(71.5, 3.4, n_batters)

    frames = []
    for year in seasons:
        n = n_per_season
        batter = rng.integers(0, n_batters, n)
        release = rng.normal(93, 5.5, n).clip(72, 104)
        desc = rng.choice(
            ["hit_into_play", "foul", "swinging_strike", "ball", "called_strike"],
            n, p=[0.26, 0.20, 0.12, 0.28, 0.14],
        )

        df = pd.DataFrame({
            "pitch_type": rng.choice(["FF", "SL", "CH", "CU", "SI", "FC"], n,
                                     p=[.35, .2, .12, .1, .18, .05]),
            "game_year": year,
            "player_name": "Pitcher, Some",      # deliberately the PITCHER, as in real data
            "batter": batter + 100_000,
            "pitcher": rng.integers(0, n_pitchers, n) + 500_000,
            "description": desc,
            "events": "",
            "des": "",
            "stand": rng.choice(["R", "L"], n, p=[.6, .4]),
            "p_throws": rng.choice(["R", "L"], n, p=[.72, .28]),
            "balls": rng.integers(0, 4, n),
            "strikes": rng.integers(0, 3, n),
            "release_speed": release,
            "plate_x": rng.normal(0, 0.78, n),
            "plate_z": rng.normal(2.4, 0.75, n),
            "sz_top": rng.normal(3.4, 0.15, n),
            "sz_bot": rng.normal(1.6, 0.12, n),
            "pfx_x": rng.normal(0, 0.9, n),
            "pfx_z": rng.normal(0.8, 0.7, n),
            "release_extension": rng.normal(6.4, 0.35, n),
            "release_spin_rate": rng.normal(2300, 300, n),
            "arm_angle": rng.normal(40, 15, n),
            "spin_axis": rng.uniform(0, 360, n),
            "effective_speed": release + rng.normal(0.5, 1.0, n),
            "swing_length": rng.normal(7.3, 0.7, n),
        })

        df["vy0"] = -(release / FPS_TO_MPH) * 0.995
        df["vx0"] = rng.normal(0, 5, n)
        df["vz0"] = rng.normal(-4, 4, n)
        df["ay"] = rng.normal(28, 3, n)
        df["ax"] = rng.normal(0, 8, n)
        df["az"] = rng.normal(-14, 6, n)

        swung = df["description"].isin(["hit_into_play", "foul", "swinging_strike"])
        df["bat_speed"] = np.where(swung, rng.normal(bat_talent[batter], 4.5), np.nan)

        t = (-df["vy0"] - np.sqrt(df["vy0"] ** 2 - 2 * df["ay"] * (50 - 17 / 12))) / df["ay"]
        plate_speed = np.sqrt(
            (df["vx0"] + df["ax"] * t) ** 2
            + (df["vy0"] + df["ay"] * t) ** 2
            + (df["vz0"] + df["az"] * t) ** 2
        ) * FPS_TO_MPH

        in_play = df["description"].eq("hit_into_play").values
        pct = (0.86
               - 0.035 * np.abs(df["plate_x"])
               - 0.030 * np.abs(df["plate_z"] - 2.5)
               - 0.0015 * (release - 92)
               + skill[batter]
               + rng.normal(0, 0.10, n))
        ev_max = 1.23 * df["bat_speed"].fillna(70) + 0.23 * plate_speed

        df["launch_speed"] = np.where(in_play, ev_max * pct.clip(0.15, 1.05), np.nan)
        df["launch_angle"] = np.where(in_play, rng.normal(12, 22, n), np.nan)
        df["estimated_woba_using_speedangle"] = np.where(
            in_play, (pct * 0.7).clip(0, 1) + rng.normal(0, 0.06, n), np.nan)
        df.loc[in_play, "events"] = "field_out"

        # bunts put in play: they arrive as hit_into_play and are invisible to
        # the bunt description codes
        ip_idx = df.index[in_play]
        if n_bunts and len(ip_idx) >= n_bunts:
            bunts = rng.choice(ip_idx, n_bunts, replace=False)
            df.loc[bunts, "bat_speed"] = rng.uniform(4, 18, n_bunts)
            df.loc[bunts, "launch_speed"] = rng.uniform(18, 42, n_bunts)
            df.loc[bunts, "events"] = "sac_bunt"
            df.loc[bunts, "des"] = "Batter out on a sacrifice bunt."

        frames.append(df)

    return pd.concat(frames, ignore_index=True)
