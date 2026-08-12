"""The expectation model: P(squared-up % | pitch, swing speed)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from .config import Config, Variant
from .features import build_features

__all__ = ["fit_expectation", "fit_all_variants"]


def _make_model(cfg: Config, categorical_mask):
    return HistGradientBoostingRegressor(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        categorical_features=categorical_mask,
        random_state=cfg.random_seed,
    )


def fit_expectation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cfg: Config,
) -> np.ndarray:
    """Out-of-fold expected squared-up percentage.

    Folds are grouped on ``batter``: every hitter's expectation comes from a
    model that never saw one of his own batted balls. Without this a skilled
    hitter inflates his own baseline and his residual collapses toward zero.

    Batter identity is never a feature -- the expectation is what a league-
    average hitter would have managed against these pitches.
    """
    categorical_mask = [dtype.name == "category" for dtype in X.dtypes]
    oof = np.full(len(X), np.nan)

    splitter = GroupKFold(n_splits=cfg.n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = _make_model(cfg, categorical_mask)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])

    return oof


def fit_all_variants(
    bip: pd.DataFrame,
    cfg: Config,
    verbose: bool = True,
) -> Dict[str, float]:
    """Fit every configured variant, writing residuals onto ``bip`` in place.

    Adds ``xpct_<variant>`` and ``resid_<variant>`` columns. Returns the
    out-of-fold R^2 per variant.

    Every variant is always fitted. Downstream code iterates ``cfg.variants``
    rather than naming one -- hard-coding a single variant into validation is
    exactly how this project once produced a false negative on itself.
    """
    y = bip["squared_up_pct"]
    groups = bip["batter"]
    scores: Dict[str, float] = {}

    for variant in cfg.variants:
        X = build_features(bip, variant)
        expected = fit_expectation(X, y, groups, cfg)
        bip[f"xpct_{variant.name}"] = expected
        bip[f"resid_{variant.name}"] = y - expected
        scores[variant.name] = float(r2_score(y, expected))
        if verbose:
            flag = " <- primary" if variant.name == cfg.primary else ""
            print(f"  {variant.name:12s} OOF R2 = {scores[variant.name]:+.4f}{flag}")

    return scores
