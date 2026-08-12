"""Command-line interface: ``python -m suax``."""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="suax",
        description="Squared-Up Above Expected: bat-speed-adjusted barrel accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  python scripts/prep_statcast.py 2024 2025\n"
            "  python -m suax --data statcast_2024_2025.parquet --out results/\n"
        ),
    )
    p.add_argument("--data", required=True,
                   help="Statcast export (.parquet strongly preferred over .csv)")
    p.add_argument("--out", default="results",
                   help="output directory for CSVs (default: results)")
    p.add_argument("--min-bbe", type=int, default=100,
                   help="batted-ball qualifier for the hitter board (default: 100)")
    p.add_argument("--min-bbe-pitcher", type=int, default=150,
                   help="batted-ball qualifier for the pitcher board (default: 150)")
    p.add_argument("--min-bbe-season", type=int, default=80,
                   help="per-season qualifier for the year-over-year test (default: 80)")
    p.add_argument("--folds", type=int, default=5, help="GroupKFold splits (default: 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-validation", action="store_true",
                   help="skip reliability, year-over-year, sweep and permutation tests")
    p.add_argument("--no-names", action="store_true",
                   help="skip the pybaseball name lookup (leaderboards show MLBAM IDs)")
    p.add_argument("--top", type=int, default=15, help="rows to print (default: 15)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(
        min_bbe=args.min_bbe,
        min_bbe_pitcher=args.min_bbe_pitcher,
        min_bbe_season=args.min_bbe_season,
        n_splits=args.folds,
        random_seed=args.seed,
    )

    try:
        result = run_pipeline(
            args.data,
            cfg,
            resolve_player_names=not args.no_names,
            run_validation=not args.no_validation,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    col = f"suax_{cfg.primary}_p100"
    show = [c for c in ["name", "bbe", "bat_speed", "su_rate", "su_pct", col]
            if c in result.hitters.columns]
    print(f"\n=== TOP {args.top} · barrel accuracy above expected ===")
    print(result.hitters.sort_values(col, ascending=False)[show]
          .head(args.top).round(3).to_string())

    result.write(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
