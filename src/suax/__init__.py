"""SUAX - Squared-Up Above Expected.

A bat-speed-adjusted measure of barrel accuracy built on Statcast bat tracking.

Statcast's published squared-up rate carries bat speed in its denominator, so
it is anti-correlated with swing speed (r = -0.55) and predicts next-season
contact quality backwards (r = -0.10). SUAX models expected squared-up
percentage from the pitch *and the hitter's bat speed*, then measures the
residual. The corrected metric predicts forward (r = +0.34).

Quickstart
----------
    from suax import Config, run_pipeline

    result = run_pipeline("statcast_2024_2025.parquet", Config())
    print(result.hitters.sort_values("suax_pitch_bs_p100", ascending=False).head())

See docs/METHODOLOGY.md for the full validation record.
"""

from .config import Config, Variant, VARIANTS, PRIMARY_VARIANT
from .physics import add_squared_up, plate_speed_mph
from .screening import tag_events, screen, league_rates, validate_columns
from .features import build_features
from .model import fit_expectation, fit_all_variants
from .aggregate import shrink, build_hitter_board, build_pitcher_board
from .pipeline import PipelineResult, run_pipeline, load_data

__version__ = "2.1.0"

__all__ = [
    "Config", "Variant", "VARIANTS", "PRIMARY_VARIANT",
    "add_squared_up", "plate_speed_mph",
    "tag_events", "screen", "league_rates", "validate_columns",
    "build_features", "fit_expectation", "fit_all_variants",
    "shrink", "build_hitter_board", "build_pitcher_board",
    "PipelineResult", "run_pipeline", "load_data",
    "__version__",
]
