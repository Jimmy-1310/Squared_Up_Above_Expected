# Squared-Up Above Expected (SUAX)

**A bat-speed-adjusted measure of barrel accuracy for MLB hitters**

Version 2.1 · Built on Statcast bat tracking, 2024–2025 · 1,443,801 pitches, 219,468 qualifying batted balls

---

## Summary

Statcast's published squared-up rate does not measure what its name implies. Because bat speed sits in the denominator of its formula, the metric is strongly anti-correlated with swing speed (r = **−0.548** across qualified hitters), and swing speed is the single best available predictor of contact quality. The result is that **raw squared-up rate predicts a hitter's next-season contact quality in the wrong direction** (r = **−0.101**).

SUAX corrects this by modelling the expected squared-up percentage of every batted ball from the pitch it came off *and the hitter's bat speed on that swing*, then measuring the residual. The corrected metric predicts next-season contact quality at r = **+0.339** — a sign flip and a swing of roughly 0.44 correlation units.

The headline practical claim: **40 batted balls of SUAX (r = +0.236) carries more usable signal than a full season of raw squared-up rate (r = −0.058).**

---

## 1. The problem

Statcast defines squared-up percentage as the share of theoretically available exit velocity a hitter obtained on a swing:

```
EV_max        = 1.23 × bat_speed + 0.23 × pitch_speed_at_plate
squared_up_%  = launch_speed / EV_max
squared up    = squared_up_% ≥ 0.80
```

The intent is sound — it isolates collision quality from raw power, so a 66 mph swing that finds the sweet spot scores as highly as an 82 mph swing that does. But it creates two problems.

**Bat speed is in the denominator.** A hitter swinging 81 mph must reach roughly 113 mph exit velocity to clear the 80% bar; a hitter swinging 63 mph needs only 92. The metric therefore ranks slow swingers highly almost by construction.

**Pitch difficulty is unmodelled.** A hanging slider middle-middle and a 99 mph fastball at the letters are scored identically.

The first problem turns out to dominate. The second is real at the pitch level but largely averages out across a season, since every qualified hitter faces a similar league-wide pitch mix.

**Observed consequence.** Across 314 hitters with 80+ batted balls in both 2024 and 2025:

| Predictor (2024) | Correlation with 2025 xwOBAcon |
|---|---|
| Bat speed | **+0.645** |
| Raw squared-up rate | **−0.101** |
| Raw squared-up % (continuous) | −0.130 |

A metric marketed as measuring contact quality is negatively associated with future contact quality. It is substantially measuring "swings slowly."

---

## 2. What SUAX measures

SUAX is the difference between the squared-up percentage a hitter actually achieved and the squared-up percentage expected given the pitches he saw and how hard he swung at them.

> **SUAX = actual squared-up % − expected squared-up %**
> where expected is modelled from pitch characteristics + bat speed

Units are percentage points of `EV_max` per batted ball. A SUAX of +2.5 means the hitter extracts 2.5 percentage points more of the available exit velocity than a league-average hitter would, facing the same pitches with the same swing speed.

Two variants are produced. Only one of them works.

| Variant | Model features | Verdict |
|---|---|---|
| `pitch_only` | Pitch velo, movement, location, count, platoon | **Fails.** Still r = −0.468 with bat speed; still predicts backwards (−0.072). |
| `pitch_bs` | Above **+ bat speed** | **Works.** r = +0.150 with bat speed; predicts forward (+0.339). |

**Use `pitch_bs`. Discard `pitch_only`.** Carrying both invites misreading — several validation tests in earlier drafts silently ran on `pitch_only` and produced a false negative verdict on the whole project.

---

## 3. Computation

**Data.** Raw Statcast pitch-level export (`pybaseball.statcast()` or Baseball Savant). Bat tracking exists from the second half of 2023 onward.

**Pitch speed at the plate.** Rather than the common `0.92 × release_speed` approximation, the trajectory is integrated from Statcast's kinematic constants. Solving `y(t) = 50 + v_y0·t + ½a_y·t²` for the front of the plate (17/12 ft) and evaluating `‖v(t)‖` there gives the true collision speed.

**Screening.** Three filters, all necessary:

1. Balls in play only. Public Statcast records `launch_speed` only for `hit_into_play`.
2. Bunts removed — including bunts *put in play*, which arrive as `hit_into_play` and are invisible to the bunt description codes. Text-scan `events` and `des`. Squared-up percentage has no bat-speed floor, so a bunt can register as "squared up."
3. Savant's competitive-swing screen: fastest 90% of a player's swings, plus any 60+ mph swing producing 90+ mph exit velocity.

**Target.** Continuous `squared_up_%`, clipped to [0.10, 1.05]. **Not** the binary flag — see §4.3.

**Model.** `HistGradientBoostingRegressor`, out-of-fold predictions via `GroupKFold` grouped on `batter`. Batter ID is never a feature. Without the grouping, a skilled hitter inflates his own baseline and his residual collapses toward zero.

Features: plate speed, handedness-mirrored horizontal location and movement, zone-normalized height, count, platoon, extension, spin rate, spin axis (as sin/cos, since it is circular), arm angle, pitch type, bat speed.

**Aggregation.** Mean residual per hitter, shrunk toward zero by empirical Bayes:

```
θ̂ᵢ = r̄ᵢ · nᵢ/(nᵢ + k),    k = σ²_within / σ²_true,
σ²_true = σ²_between − mean(σ²_within / nᵢ)
```

Fitted k = **73.7** batted balls for the `pitch_bs` variant — the sample size at which observed performance and league average are weighted equally.

---

## 4. Validation

### 4.1 Physics reconstruction

Independent reconstruction from raw kinematics, checked against MLB's published figures:

| Quantity | Computed | Published |
|---|---|---|
| Squared-up per competitive swing | **25.1%** | ~25% |
| Mean bat speed | **72.0 mph** | ~71.5 mph |
| Squared-up per ball in play | 66.4% | not published |

The 25.1% match to within 0.1 points confirms the physics layer.

> **Denominator warning.** MLB's widely-quoted "33% of contacts" figure uses a denominator of *all contact including fouls*. Public Statcast has no `launch_speed` on fouls, so that number is not reproducible from public data. Comparing an in-play rate against 33% will make a correct pipeline look broken. Expect 60–70% in play.

### 4.2 Face validity

Top of the **raw** squared-up leaderboard: Luis Arráez (.845), Edgar Quero (.825), Tyler Nevin (.805), Alex Call (.791), Josh Rojas (.786), Luis Guillorme (.786), Michael Busch (.785), Steven Kwan (.781). Exactly the contact-specialist profile expected.

Top of the **adjusted** leaderboard: Ohtani, Devers, Oneil Cruz, Arráez, Judge, Busch, Soto, Aranda, Yandy Díaz, Bichette. Bottom: Fraley, Stubbs, Connor Joe, Varsho, Biggio, McGuire, Fortes, Heyward, Rortvedt, Mayo — heavily catchers and weak-contact bats.

**The illustrative case:**

| Hitter | Bat speed | Raw squared-up | SUAX (pitch only) | SUAX (+bat speed) |
|---|---|---|---|---|
| Giancarlo Stanton | 81.4 | .588 | **−1.92** | **+2.09** |
| Luis Arráez | 63.4 | .845 | +5.15 | +2.78 |

Unadjusted, Stanton looks like one of the league's worst barrel-finders and Arráez looks superhuman — a 26-point gap. Adjusted for the fact that an 81.4 mph swing sets a far higher absolute bar, the gap collapses to under one point. Both are good at this; the raw metric simply could not see Stanton.

### 4.3 Reliability

Split-half, 25 random splits, Spearman–Brown corrected:

| Metric | Reliability |
|---|---|
| Raw squared-up rate (binary) | 0.739 |
| Raw squared-up % (continuous) | 0.816 |
| SUAX, binary target *(v1)* | 0.713 |
| **SUAX, continuous target, pitch only** | **0.845** |
| **SUAX, continuous target, +bat speed** | **0.811** |

Moving from the binary flag to the continuous target lifted reliability from 0.713 to 0.845. Thresholding a smooth variable at 0.80 discards the distinction between a 0.79 and a 0.35, and imposes a binomial variance floor of `p(1−p)/n`. **This was the single largest methodological error in v1 and its correction changed the project's conclusions in two places** — hitter reliability here, and the pitcher result in §4.7.

Year-over-year (2024 → 2025, n = 314): bat speed 0.932, raw squared-up rate 0.652, SUAX (+bat speed) 0.665. All of these are genuine, moderately stable traits.

### 4.4 Decoupling from bat speed

The direct test of whether the denominator artifact was removed:

| Metric | Correlation with bat speed |
|---|---|
| Raw squared-up rate | **−0.548** |
| SUAX, pitch only | −0.468 |
| **SUAX, + bat speed** | **+0.150** |

Pitch-difficulty adjustment alone barely moves the artifact. Only entering bat speed into the expectation model removes it.

### 4.5 Predictive validity

Predicting 2025 xwOBAcon from 2024 metrics (n = 314):

| Predictor | r |
|---|---|
| Bat speed | +0.645 |
| Raw squared-up rate | −0.101 |
| SUAX, pitch only | −0.072 |
| **SUAX, + bat speed** | **+0.339** |

Nested out-of-sample R²:

| Model | R² | ΔR² |
|---|---|---|
| Bat speed | 0.4160 | — |
| + raw squared-up rate | 0.5074 | +0.0914 |
| + SUAX (pitch only) | 0.5082 | +0.0008 |
| + SUAX (+bat speed) | 0.5107 | +0.0024 |

Parsimonious comparison:

| Model | R² | ΔR² |
|---|---|---|
| Bat speed alone | 0.4160 | — |
| Bat speed + SUAX (+bat speed) | 0.4758 | **+0.0598** |

**Read these two tables together — they say different things and the difference matters.**

SUAX adds +0.060 R² over bat speed alone. But it adds only +0.0024 once raw squared-up rate is *already in the regression*. That is because a multivariate model containing both bat speed and raw squared-up rate can undo the denominator artifact linearly on its own — the regression discovers the correction that SUAX bakes in.

**So the honest claim is narrower than "SUAX is a better metric."** If you are already fitting a multivariate model, `bat_speed + su_rate` (R² = 0.507) marginally beats `bat_speed + SUAX` (R² = 0.476). SUAX's value is as a **standalone, directionally correct, interpretable number** — for a leaderboard, a scouting report, a player card, or any context where one figure must stand on its own. In that role raw squared-up rate is not merely weaker, it is *backwards*, and SUAX is not.

### 4.6 Sample-size behaviour

Correlation with 2025 xwOBAcon, by 2024 subsample size:

| n batted balls | Raw squared-up | SUAX (pitch only) | **SUAX (+bat speed)** |
|---|---|---|---|
| 40 | −0.066 | −0.055 | **+0.236** |
| 80 | −0.085 | −0.060 | **+0.283** |
| 150 | −0.114 | −0.085 | **+0.328** |
| 300 | −0.058 | −0.037 | **+0.468** |

**A prior hypothesis was disconfirmed here.** We expected the adjustment's advantage to *widen* as samples shrank, on the theory that pitch difficulty averages out over a full season but not over 80 batted balls. The opposite occurred: the advantage grows with sample size (gap of 0.30 at n=40, 0.53 at n=300). SUAX is not a small-sample correction. It is a **bias correction that operates at every sample size**, because the bias it removes — bat speed in the denominator — is present in every batted ball.

The practical implication is stronger than the original hypothesis would have been: SUAX at 40 batted balls (+0.236) already carries more signal than raw squared-up rate at 300 (−0.058). The raw metric is not weak in small samples; it is unusable at any sample size.

### 4.7 Pitcher side

The same residuals aggregate by pitcher. A permutation test — shuffling residuals across pitchers while holding each pitcher's batted-ball count fixed, 400 iterations — tests whether observed spread exceeds pure sampling noise.

| Group | Observed variance | Null mean | p |
|---|---|---|---|
| Pitchers (n=478) | 4.889e-05 | 2.980e-05 | **0.0000** |
| Hitters (n=498, control) | 1.978e-04 | 2.786e-05 | **0.0000** |

**Real pitcher-level skill exists.** The hitter control confirms the test has power; pitchers clear the null independently. Best contact suppressors: Freddy Peralta, Mike King, Hunter Brown, Kris Bubic, Andrew Abbott, Tarik Skubal, Chris Sale, Bryan Woo — a face-valid list including two Cy Young winners.

The effect is modest: implied true SD ≈ 0.44 percentage points, and fitted k = 1194 batted balls means heavy shrinkage. **v1 concluded the opposite** — no detectable pitcher signal — purely because the binary target's noise floor swamped a real but small effect.

---

## 5. How to use it

### Where SUAX is the right tool

**Standalone hitter evaluation.** Any single-number context — leaderboards, player cards, scouting summaries, org rankings. Raw squared-up rate points the wrong way here and should not be displayed without a bat-speed adjustment alongside it.

**Small-sample barrel-skill reads.** 40+ batted balls gives a usable signal (r ≈ +0.24). Relevant for prospects, call-ups, post-injury returns, mid-season swing changes, and trade-deadline evaluation on partial data.

**Separating two skills that raw metrics confound.** Bat speed and barrel accuracy are near-orthogonal after adjustment (r = +0.150), so plotting them as two axes cleanly separates power from precision. Hitters in the high-both quadrant (Ohtani, Judge, Soto) are distinguishable from high-power/low-precision and low-power/high-precision profiles in a way raw squared-up rate cannot support.

**Pitcher contact suppression.** Modest but statistically real. Useful as a supplementary axis alongside stuff models, not as a headline number.

### Where SUAX is *not* the right tool

**Inside a multivariate model that already contains bat speed and squared-up rate.** The regression performs the correction itself; SUAX adds +0.0024. Use the raw components.

**As a replacement for bat speed.** Bat speed alone explains R² = 0.416 of next-season contact quality; SUAX adds +0.060 to that. It is a complement, not a substitute.

**Anywhere foul-ball behaviour matters.** See §6.

### Reading the numbers

Units are percentage points of available exit velocity per batted ball. Qualified range runs roughly −4.6 (Fraley) to +3.6 (Ohtani). Shrinkage is already applied; k = 73.7 means a hitter with 74 batted balls is weighted 50/50 against league average.

---

## 6. Limitations

**Fouls are invisible.** Public Statcast records `launch_speed` only for balls in play, so SUAX measures barrel accuracy *conditional on putting the ball in play*. A hitter who fouls off pitches he cannot barrel is flattered. This also means the published "33% of contacts" benchmark is not reproducible from public data.

**Bat speed is measured at contact,** so it is partly a timing artifact — contact further in front means more distance to accelerate. Since bat speed is both a model feature and part of the denominator, this noise enters twice. Adding `intercept_ball_minus_batter_pos_y_inches` as a feature would absorb some of it; untested.

**Sequencing is unmodelled.** Every pitch is treated independently. Previous-pitch type and velocity differential are the obvious extension and the most plausible route to strengthening the pitcher side.

**Pitcher-side confounds.** Pitchers do not face random hitters. Platoon usage, opponent quality, and pitching half of one's innings in a single park all induce within-pitcher correlation that is not skill. Opponent-hitter and park fixed effects would tighten §4.7 considerably. The current result should be read as suggestive, not settled.

**The EV_max formula ignores bat mass and moment of inertia.** `1.23·bat_speed + 0.23·pitch_speed` is Statcast's published approximation. Hitters on unusual bat profiles — torpedo bats especially, where the entire design premise is relocating mass distribution — will be systematically mis-scored.

**Two seasons only.** Year-over-year results rest on a single 2024→2025 transition with n = 314. Additional seasons would firm up every predictive figure here.

---

## 7. Files

| File | Purpose |
|---|---|
| `SUAX_v2_squared_up_above_expected.ipynb` | Full pipeline: physics, screening, modelling, all validation |
| `SUAX_v2_corrections.py` | Corrected nested ΔR² and sample-size sweep (append as final cells) |
| `prep_statcast.py` | Pulls Statcast season-by-season, caches to parquet, verifies integrity |
| `suax_v2_hitters.csv` | Hitter leaderboard |
| `suax_v2_pitchers.csv` | Pitcher leaderboard |
| `suax_v2_yoy.csv` | Year-over-year paired seasons |
| `suax_v2_pitch_level.csv` | Per-batted-ball residuals |

**Reproduction note.** Do not upload large CSVs through Colab's `files.upload()` — it silently truncates and produces an empty leaderboard downstream. Cache to parquet via `prep_statcast.py` and set `DATA_PATH`.

---

## Appendix: what changed between versions

| Issue | v1 | v2 | Consequence |
|---|---|---|---|
| Batter labels | Used `player_name` | `playerid_reverse_lookup` on `batter` | `player_name` is the **pitcher** in raw Statcast. v1's leaderboard listed relievers as top hitters and no face-validity check was possible. |
| Target | Binary flag at 0.80 | Continuous `squared_up_%` | Reliability 0.713 → 0.845. Also flipped the pitcher conclusion from "no signal" to p < 0.0001. |
| Validation | Same-season R² | Year-over-year | Same-season xwOBAcon is exit-velocity-derived, so squared-up rate was partly predicting itself (ΔR² = +0.15 was spurious). |
| Bunts | Description codes only | Text-scan `events`/`des` | Bunts put in play arrive as `hit_into_play` and were contaminating the sample. |
| Which variant tested | `pitch_only` hard-coded | Both | The working variant (`pitch_bs`) was never tested in v2's first run, producing a false negative on the entire project. |

The two conclusions that reversed between versions — hitter reliability and pitcher signal — both traced to the same root cause: binarizing a continuous target.
