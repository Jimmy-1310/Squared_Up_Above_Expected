"""Configuration for the SUAX pipeline.

Every tunable lives here. Nothing downstream hard-codes a threshold, a
coefficient, or a variant name -- an earlier iteration of this project
hard-coded one variant into two validation tests and produced a false
negative on the entire metric. See docs/METHODOLOGY.md, appendix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# --- Statcast's published squared-up formula ---------------------------------
# EV_max = EVMAX_BAT_COEF * bat_speed + EVMAX_PITCH_COEF * pitch_speed_at_plate
EVMAX_BAT_COEF = 1.23
EVMAX_PITCH_COEF = 0.23          # Clemens' published variant uses 0.2116
SQUARED_UP_THRESHOLD = 0.80

# --- geometry / units --------------------------------------------------------
PLATE_FRONT_Y = 17.0 / 12.0      # front of home plate, feet from the tip
STATCAST_Y0 = 50.0               # y at which vx0/vy0/vz0 are reported
FPS_TO_MPH = 0.681818

# --- pitch descriptions ------------------------------------------------------
SWING_DESCRIPTIONS = frozenset({
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "foul", "foul_tip", "swinging_strike", "swinging_strike_blocked",
})
BUNT_DESCRIPTIONS = frozenset({"foul_bunt", "missed_bunt", "bunt_foul_tip"})
IN_PLAY_DESCRIPTIONS = frozenset({
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
})

REQUIRED_COLUMNS = (
    "description", "batter", "pitcher", "stand", "p_throws", "bat_speed",
    "launch_speed", "release_speed", "plate_x", "plate_z", "pfx_x", "pfx_z",
    "sz_top", "sz_bot", "balls", "strikes", "pitch_type",
)

KINEMATIC_COLUMNS = ("vx0", "vy0", "vz0", "ax", "ay", "az")

OPTIONAL_FEATURES = (
    "release_extension", "release_spin_rate", "arm_angle", "effective_speed",
    "api_break_z_with_gravity", "api_break_x_batter_in",
)

SWING_PATH_FEATURES = (
    "attack_angle", "swing_path_tilt", "attack_direction",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
)


@dataclass(frozen=True)
class Variant:
    """One expectation-model specification.

    ``include_bat_speed`` is the axis that matters. Without it the residual
    stays correlated with bat speed (r = -0.47) and predicts next-season
    contact quality backwards (r = -0.07). With it: r = +0.15 and +0.34.

    ``include_swing_path`` is off by default. Attack angle and intercept point
    describe the *swing*, not the pitch, so including them partially explains
    away the very skill being measured. Available for deliberate experiments.
    """

    name: str
    include_bat_speed: bool
    include_swing_path: bool = False
    note: str = ""


#: ``pitch_bs`` is the metric. ``pitch_only`` is retained only as the
#: documented failed baseline, so that validation always reports both and
#: nobody silently tests the wrong one again.
VARIANTS: Tuple[Variant, ...] = (
    Variant("pitch_only", include_bat_speed=False,
            note="FAILED BASELINE - does not remove the bat-speed artifact"),
    Variant("pitch_bs", include_bat_speed=True,
            note="THE METRIC - use this one"),
)
PRIMARY_VARIANT = "pitch_bs"


@dataclass
class Config:
    """Runtime settings for a SUAX run."""

    # screening
    apply_competitive_filter: bool = True
    drop_bunts_by_text: bool = True
    clip_pct: Tuple[float, float] = (0.10, 1.05)

    # modelling
    n_splits: int = 5
    random_seed: int = 42
    max_iter: int = 500
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0

    # qualifiers
    min_bbe: int = 100            # season leaderboard
    min_bbe_season: int = 80      # per-season, for the year-over-year test
    min_bbe_pitcher: int = 150    # pitcher leaderboard

    # validation
    split_half_iters: int = 25
    permutation_iters: int = 400
    sweep_sizes: Tuple[int, ...] = (40, 80, 150, 300)
    sweep_reps: int = 12

    variants: Tuple[Variant, ...] = field(default=VARIANTS)
    primary: str = PRIMARY_VARIANT

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        known = [v.name for v in self.variants]
        raise KeyError(f"unknown variant {name!r}; have {known}")
