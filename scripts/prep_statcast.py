"""
Pull Statcast pitch-level data and cache it as parquet.

Run this ONCE per season. After that the notebook loads in seconds and you
never touch a browser upload again.

    pip install pybaseball pandas pyarrow
    python prep_statcast.py 2024 2025

Then in the notebook:

    DATA_PATH = "statcast_2024_2025.parquet"

Why parquet: a full Statcast season is ~700k rows x ~120 columns. As CSV that
is roughly 700-900 MB per season and takes minutes to parse. Parquet is
columnar and compressed - typically 5-10x smaller and far faster to load.
Bat tracking (bat_speed, swing_length) exists only from the second half of
2023 onward, so earlier seasons will come back without those columns.
"""

import sys
import os
import time

import pandas as pd

try:
    from pybaseball import statcast, cache
except ImportError:
    sys.exit("pip install pybaseball pyarrow")

# pybaseball's own on-disk cache: avoids re-downloading if you rerun
cache.enable()

# Statcast rejects very large date ranges, so pull in chunks and concatenate.
CHUNK_DAYS = 14

SEASON_BOUNDS = {
    # regular season roughly; widen if you want spring/postseason
    "start": "-03-15",
    "end": "-11-15",
}


def pull_season(year: int) -> pd.DataFrame:
    start = f"{year}{SEASON_BOUNDS['start']}"
    end = f"{year}{SEASON_BOUNDS['end']}"
    print(f"\n=== {year}: {start} -> {end} ===")
    t0 = time.time()

    dates = pd.date_range(start, end, freq=f"{CHUNK_DAYS}D")
    frames = []
    for i, d0 in enumerate(dates):
        d1 = min(d0 + pd.Timedelta(days=CHUNK_DAYS - 1), pd.Timestamp(end))
        try:
            chunk = statcast(
                start_dt=d0.strftime("%Y-%m-%d"),
                end_dt=d1.strftime("%Y-%m-%d"),
                verbose=False,
            )
        except Exception as e:                      # network hiccup, empty window
            print(f"  [{i+1}/{len(dates)}] {d0.date()} failed: {e}")
            continue
        if chunk is not None and len(chunk):
            frames.append(chunk)
            print(f"  [{i+1}/{len(dates)}] {d0.date()}  {len(chunk):>7,} pitches")

    if not frames:
        raise RuntimeError(f"no data returned for {year}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    print(f"  {year} total: {len(df):,} pitches in {time.time()-t0:.0f}s")
    return df


def verify(df: pd.DataFrame) -> None:
    """Catch a truncated or malformed pull before it reaches the notebook."""
    print("\n=== INTEGRITY ===")
    n_years = df["game_year"].nunique()
    print(f"rows: {len(df):,} across {n_years} season(s)")
    if len(df) < 400_000 * n_years:
        print("  *** WARNING: well below ~700k pitches/season. Pull looks incomplete. ***")

    if "bat_speed" not in df.columns:
        print("  *** WARNING: no bat_speed column. Bat tracking starts mid-2023. ***")
        return

    cov = df["bat_speed"].notna().mean()
    print(f"bat_speed coverage: {cov:.1%}  (expect ~45% - it exists only on swings)")

    swings = df["description"].isin(
        ["hit_into_play", "foul", "foul_tip", "swinging_strike", "swinging_strike_blocked"]
    )
    bs = df.loc[swings, "bat_speed"].mean()
    print(f"mean bat speed on swings: {bs:.1f} mph  (MLB ~71.5)")
    if not (68 < bs < 75):
        print("  *** WARNING: bat speed far from league average. Check the data. ***")

    for c in ["vx0", "vy0", "ax", "ay", "launch_speed", "plate_x", "plate_z",
              "attack_angle", "intercept_ball_minus_batter_pos_y_inches"]:
        flag = "ok " if c in df.columns else "MISSING"
        print(f"  {flag} {c}")


def main() -> None:
    years = [int(a) for a in sys.argv[1:]] or [2024, 2025]
    frames = [pull_season(y) for y in years]
    df = pd.concat(frames, ignore_index=True)

    verify(df)

    tag = f"{min(years)}_{max(years)}" if len(years) > 1 else str(years[0])
    out = f"statcast_{tag}.parquet"
    df.to_parquet(out, compression="snappy", index=False)

    mb = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({mb:,.0f} MB, {len(df):,} rows)")
    print(f'\nIn the notebook set:\n    DATA_PATH = "{os.path.abspath(out)}"')


if __name__ == "__main__":
    main()
