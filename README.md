# SUAX — Squared-Up Above Expected

**Statcast's squared-up rate predicts a hitter's future contact quality backwards. This fixes it.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The problem

Statcast defines squared-up percentage as the share of theoretically available exit velocity a hitter obtains:

```
EV_max       = 1.23 × bat_speed + 0.23 × pitch_speed_at_plate
squared_up_% = launch_speed / EV_max
squared up   = squared_up_% ≥ 0.80
```

Bat speed sits in the **denominator**. A hitter swinging 81 mph must reach ~113 mph exit velocity to clear the bar; a hitter swinging 63 mph needs only 92. The metric therefore rewards slow swingers almost by construction — and bat speed is the single best available predictor of contact quality.

The consequence, measured across 314 hitters with 80+ batted balls in both 2024 and 2025:

| 2024 predictor | Correlation with 2025 xwOBAcon |
| --- | --- |
| Bat speed | **+0.645** |
| **Raw squared-up rate** | **−0.101** |
| **SUAX (bat-speed adjusted)** | **+0.339** |

A metric marketed as measuring contact quality is *negatively* associated with future contact quality. It is substantially measuring "swings slowly."

## What SUAX does

Models the expected squared-up percentage of every batted ball from the pitch it came off **and the hitter's bat speed on that swing**, then measures the residual.

> **SUAX = actual squared-up % − expected squared-up %**

Units are percentage points of `EV_max` per batted ball. Qualified 2024–25 range runs about −4.6 to +3.6.

**The illustrative case:**

| Hitter | Bat speed | Raw squared-up | SUAX |
| --- | --- | --- | --- |
| Giancarlo Stanton | 81.4 | .588 | **+2.09** |
| Luis Arráez | 63.4 | .845 | **+2.78** |

Unadjusted, Stanton looks like one of the league's worst barrel-finders and Arráez looks superhuman — a 26-point gap. Adjusted for the fact that an 81.4 mph swing sets a far higher absolute bar, the gap collapses to under one point. Both are good at this. The raw metric simply could not see Stanton.

## Install

```bash
git clone https://github.com/Jimmy-1310/suax.git
cd suax
pip install -e ".[data]"     # [data] adds pybaseball for pulling + name lookup
```

## Quickstart

```bash
# 1. Pull and cache the data (once per season). Parquet, not CSV.
python scripts/prep_statcast.py 2024 2025

# 2. Run
python -m suax --data statcast_2024_2025.parquet --out results/
```

Or as a library:

```python
from suax import Config, run_pipeline

result = run_pipeline("statcast_2024_2025.parquet", Config())

print(result.hitters.sort_values("suax_pitch_bs_p100", ascending=False).head(15))
print(result.reliability)
print(result.spread_tests["pitchers"])

result.write("results/")
```

> **Do not upload large CSVs through Colab's `files.upload()`.** It silently truncates, and the failure surfaces far downstream as an empty leaderboard. Use `scripts/prep_statcast.py` to cache parquet — a two-season pull is ~1.5 GB as CSV and a few hundred MB as parquet.

## Validation

Full record in [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Headlines from 1,443,801 pitches across 2024–25 (219,468 qualifying batted balls, 498 qualified hitters):

**Physics reconstruction.** Independently reconstructed squared-up per competitive swing at **25.1%** against MLB's published ~25%, and mean bat speed at 72.0 mph against ~71.5. The physics layer is not in question.

**Face validity.** Top of the raw leaderboard: Arráez, Quero, Nevin, Call, Rojas, Guillorme, Busch, Kwan — exactly the contact-specialist profile. Top adjusted: Ohtani, Devers, Oneil Cruz, Arráez, Judge, Soto.

**Decoupling.** Correlation with bat speed goes from **−0.548** (raw) to **+0.150** (adjusted).

**Reliability.** Split-half Spearman–Brown **0.811**; year-over-year **0.665**.

**Sample-size behaviour.** SUAX is positive at every sample size tested; raw squared-up rate is negative at every one.

| n batted balls | Raw | **SUAX** |
| --- | --- | --- |
| 40 | −0.066 | **+0.236** |
| 80 | −0.085 | **+0.283** |
| 150 | −0.114 | **+0.328** |
| 300 | −0.058 | **+0.468** |

**40 batted balls of SUAX carries more signal than a full season of raw squared-up rate.**

**Pitcher side.** A permutation test (shuffling residuals across pitchers while holding batted-ball counts fixed) rejects the null at p < 0.0001, with hitters as a positive control. Best contact suppressors: Peralta, Mike King, Hunter Brown, Bubic, Abbott, Skubal, Sale, Woo. The effect is modest (implied true SD ≈ 0.44 points) and confounded by non-random opponent and park assignment — read it as suggestive, not settled.

## What SUAX is *not*

This is documented as carefully as what it is, because the distinction determines where it should be used.

**It does not beat raw squared-up rate inside a multivariate model.** Predicting next-season xwOBAcon:

| Model | R² |
| --- | --- |
| Bat speed alone | 0.416 |
| Bat speed + SUAX | 0.476 |
| Bat speed + raw squared-up rate | **0.507** |

A regression containing both bat speed and raw squared-up rate can undo the denominator artifact linearly on its own — it rediscovers the correction SUAX bakes in. Adding SUAX on top of both is worth only +0.0024.

**So the claim is narrower than "SUAX is a better metric."** Its value is as a **standalone, directionally correct, interpretable number** — a leaderboard, a scouting report, a player card, anywhere one figure must stand alone. In that role raw squared-up rate is not merely weaker; it points the wrong way, and SUAX does not.

**A prior hypothesis was also disconfirmed.** We expected the adjustment to matter most in small samples, since pitch difficulty averages out over a season. The opposite held — the edge *grows* with sample size. SUAX is not a small-sample correction; it is a bias correction operating at every *n*, because the bias is in every batted ball.

## How it works

```
prep_statcast.py  →  parquet cache
        ↓
physics.py        →  integrate the trajectory for true plate speed, compute squared_up_%
screening.py      →  drop bunts (including in-play bunts), apply the competitive-swing rule
features.py       →  mirror by handedness, normalise the zone, encode spin axis circularly
model.py          →  HistGradientBoostingRegressor, GroupKFold on batter, out-of-fold
aggregate.py      →  empirical-Bayes shrinkage (k ≈ 74 batted balls)
validation.py     →  reliability · year-over-year · sample sweep · permutation test
```

Three design choices carry most of the weight:

- **Continuous target, not the 0.80 binary.** Thresholding a smooth variable discards the difference between 0.79 and 0.35, and imposes a binomial variance floor. Switching to continuous lifted reliability from 0.713 to 0.845 — and flipped the pitcher conclusion from "no signal" to p < 0.0001.
- **Out-of-fold expectations grouped on `batter`.** Batter identity is never a feature. Without the grouping a skilled hitter inflates his own baseline and his residual collapses toward zero.
- **Trajectory-integrated plate speed.** Solve `y(t) = 50 + v_y0·t + ½a_y·t²` for the front of the plate rather than approximating with `0.92 × release_speed`.

## Known limitations

- **Fouls are invisible.** Public Statcast records `launch_speed` only for balls in play, so this measures barrel accuracy *conditional on putting the ball in play*. A hitter who fouls off what he cannot barrel is flattered. This also means MLB's published "33% of contacts" figure is not reproducible from public data — expect 60–70% in play, and do not treat the gap as a bug.
- **Bat speed is measured at contact,** so it is partly a timing artifact — contact further in front means more distance to accelerate. That noise enters both the model features and the denominator.
- **Sequencing is unmodelled.** Every pitch is treated independently. Previous-pitch type and velocity differential are the obvious extension.
- **Pitcher-side confounds.** Pitchers do not face random hitters. Platoon usage, opponent quality, and park all induce within-pitcher correlation that is not skill.
- **The EV_max formula ignores bat mass and moment of inertia.** Hitters on unusual bat profiles — torpedo bats especially — will be systematically mis-scored.
- **Two seasons only.** Every predictive figure rests on a single 2024→2025 transition, n = 314.

## Repository layout

```
src/suax/
  config.py       all tunables, variant definitions
  physics.py      plate speed integration, EV_max, squared_up_%
  screening.py    swing tagging, bunt removal, competitive-swing filter
  features.py     feature construction
  model.py        out-of-fold expectation model
  aggregate.py    empirical-Bayes shrinkage, leaderboards
  validation.py   reliability, year-over-year, sweep, permutation test
  names.py        MLBAM ID → name (player_name is the PITCHER — see below)
  pipeline.py     orchestration
  cli.py          python -m suax
scripts/prep_statcast.py     pull + cache + integrity-check
notebooks/suax_exploration.ipynb   the original exploratory notebook
docs/METHODOLOGY.md          full methodology and validation record
tests/                       19 tests, several of them regression tests for real bugs
```

## Development

```bash
pip install -e ".[dev]"
pytest                      # 19 tests, ~30s
pytest -m "not slow"        # skip the model-fitting end-to-end tests
```

Tests run entirely on synthetic Statcast-shaped data with injected ground truth (`tests/synthetic.py`) — no network, no credentials, no real data required.

## Bugs this codebase encodes tests against

Every one of these shipped in an earlier version and produced a wrong answer that looked plausible:

| Bug | Symptom | Guard |
| --- | --- | --- |
| `player_name` is the **pitcher**, not the batter | Hitter leaderboard made entirely of relievers | `names.resolve_names` on MLBAM IDs |
| Bunts put in play arrive as `hit_into_play` | Squared-up rate inflated; bat speeds under 20 mph in the sample | `test_bunts_put_in_play_are_removed` |
| Binarising the target at 0.80 | Reliability 0.713 instead of 0.845; pitcher signal invisible | Continuous target by default |
| Same-season validation | xwOBAcon is exit-velocity-derived, so squared-up rate partly predicted itself | Year-over-year only |
| Hard-coding one variant into validation | The working variant was never tested; false negative on the whole project | Validation iterates `cfg.variants`; `test_pipeline_end_to_end` asserts every variant appears |
| Empty leaderboard from a truncated upload | Opaque `ZeroDivisionError` from numpy | `test_shrink_raises_on_empty_input` + qualification table |

## Citation

```bibtex
@software{castillo_suax_2026,
  author  = {Castillo, Jaime},
  title   = {SUAX: Squared-Up Above Expected},
  year    = {2026},
  url     = {https://github.com/Jimmy-1310/suax}
}
```

Built on MLB Statcast bat tracking data via [pybaseball](https://github.com/jldbc/pybaseball). Not affiliated with or endorsed by MLB.

## License

MIT — see [LICENSE](LICENSE).
