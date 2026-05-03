"""
Live ML inference — load a trained LightGBM artifact bundle, build features
for an as-of date, predict calibrated 3-class probabilities per symbol,
and map them to BUY / HOLD / SELL recommendations.

Pairs with `quant.ml.trainer` (which now persists per-fold boosters +
isotonic calibrators + the exact feature column order). Loading the bundle
back is deterministic; predictions for the same inputs are byte-stable
modulo float64 ULP drift in LightGBM's internal sums.

Recommendation thresholds:
    BUY   if (P(+1)_cal - P(-1)_cal) >  +0.10
    SELL  if (P(+1)_cal - P(-1)_cal) <  -0.10
    HOLD  otherwise

The 0.10 threshold matches roughly one in-sample calibrated-ECE
standard deviation — moves below that are noise.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from quant.features.technical import add_technical_features
from quant.ml.calibration import apply_calibrators

log = logging.getLogger("quant.ml.predict")

# Class order: matches trainer._CLASSES = (-1, 0, +1)
_CLASS_NEG = 0
_CLASS_POS = 2

DEFAULT_DECISION_THRESHOLD = 0.10


@dataclass(frozen=True)
class ModelBundle:
    """Loaded artifact bundle — boosters, calibrators, feature schema."""

    boosters: list[lgb.Booster]
    calibrators: list[IsotonicRegression]
    feature_names: list[str]
    classes: tuple[int, ...]
    artifact_dir: Path


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed SHAP contribution to a single prediction."""

    feature: str
    value: float
    contribution: float  # signed; positive pushes toward BUY, negative toward SELL


@dataclass(frozen=True)
class Recommendation:
    """Per-symbol recommendation with calibrated probabilities."""

    symbol: str
    as_of: date
    prob_neg1: float
    prob_zero: float
    prob_pos1: float
    score: float  # P(+1) - P(-1), in [-1, 1]
    action: str  # "BUY" | "HOLD" | "SELL"
    confidence: str  # "low" | "medium" | "high"
    top_drivers: list[FeatureContribution] | None = None  # top SHAP features


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_bundle(artifact_dir: str | Path) -> ModelBundle:
    """Reload a trainer artifact bundle. Raises if any piece is missing."""
    p = Path(artifact_dir)
    boosters_dir = p / "boosters"
    if not boosters_dir.is_dir():
        raise FileNotFoundError(f"missing boosters dir: {boosters_dir}")
    booster_files = sorted(boosters_dir.glob("fold_*.txt"))
    if not booster_files:
        raise FileNotFoundError(f"no fold_*.txt boosters in {boosters_dir}")
    boosters = [lgb.Booster(model_file=str(f)) for f in booster_files]

    cal_path = p / "calibrators.pkl"
    if not cal_path.is_file():
        raise FileNotFoundError(f"missing calibrators.pkl: {cal_path}")
    with cal_path.open("rb") as fh:
        # Calibrators are sklearn IsotonicRegression instances written by
        # our own trainer. The artifact bundle is treated as trusted (same
        # provenance as the booster files); pickle.load is acceptable here.
        calibrators = pickle.load(fh)  # noqa: S301
    if not isinstance(calibrators, list):
        raise ValueError(f"unexpected calibrators payload in {cal_path}")

    meta_path = p / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing model_meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = list(meta["feature_names"])
    classes = tuple(int(c) for c in meta["classes"])

    log.info(
        "loaded bundle from %s — %d boosters, %d calibrators, %d features",
        p,
        len(boosters),
        len(calibrators),
        len(feature_names),
    )
    return ModelBundle(
        boosters=boosters,
        calibrators=calibrators,
        feature_names=feature_names,
        classes=classes,
        artifact_dir=p,
    )


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
def predict_calibrated(
    bundle: ModelBundle,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Average raw probabilities across folds, then apply isotonic calibration.
    Returns (n_samples, n_classes) array of calibrated probabilities.
    """
    if feature_matrix.ndim != 2:
        raise ValueError(f"feature_matrix must be 2-D, got shape {feature_matrix.shape}")

    raw_sum = np.zeros((feature_matrix.shape[0], len(bundle.classes)), dtype=np.float64)
    for booster in bundle.boosters:
        raw = booster.predict(feature_matrix, num_iteration=booster.best_iteration)
        raw_sum += np.asarray(raw, dtype=np.float64)
    raw_mean = raw_sum / len(bundle.boosters)
    return apply_calibrators(raw_mean, bundle.calibrators)


def shap_score_contributions(
    bundle: ModelBundle,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Per-row, per-feature TreeSHAP contributions to the BUY-minus-SELL score
    (P(+1) - P(-1)) on the *raw* logit space, averaged across folds.

    Returns (n_samples, n_features). Positive values push the symbol toward
    BUY; negative values push toward SELL. Bias terms are dropped because
    they are constants per fold and don't help interpretation.

    Note: LightGBM's `pred_contrib=True` returns log-odds contributions for
    each class. We compute pos_class_contrib - neg_class_contrib and
    average across folds. This matches the score the recommendation
    policy uses, modulo the calibration step (calibration is monotone in
    each class score; signs of the contributions are preserved).
    """
    if feature_matrix.ndim != 2:
        raise ValueError(f"feature_matrix must be 2-D, got shape {feature_matrix.shape}")
    n_features = feature_matrix.shape[1]
    accum = np.zeros((feature_matrix.shape[0], n_features), dtype=np.float64)
    for booster in bundle.boosters:
        contrib = booster.predict(
            feature_matrix,
            num_iteration=booster.best_iteration,
            pred_contrib=True,
        )
        contrib_arr = np.asarray(contrib, dtype=np.float64)
        # Reshape to (n_samples, n_classes, n_features+1).
        n_classes = len(bundle.classes)
        contrib_arr = contrib_arr.reshape(feature_matrix.shape[0], n_classes, n_features + 1)
        # Drop the last column (bias) and take pos_class - neg_class.
        contrib_no_bias = contrib_arr[:, :, :n_features]
        diff = contrib_no_bias[:, _CLASS_POS, :] - contrib_no_bias[:, _CLASS_NEG, :]
        accum += diff
    return accum / len(bundle.boosters)


def top_drivers(
    contributions: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    *,
    k: int = 5,
) -> list[list[FeatureContribution]]:
    """
    Per-row, top-K features ranked by |contribution| desc.
    `contributions` and `feature_values` must be aligned (n_samples × n_features).
    """
    if contributions.shape != feature_values.shape:
        raise ValueError(f"contribution shape {contributions.shape} != feature shape {feature_values.shape}")
    out: list[list[FeatureContribution]] = []
    for i in range(contributions.shape[0]):
        order = np.argsort(-np.abs(contributions[i]))[:k]
        row = [
            FeatureContribution(
                feature=feature_names[int(j)],
                value=float(feature_values[i, int(j)]),
                contribution=float(contributions[i, int(j)]),
            )
            for j in order
        ]
        out.append(row)
    return out


def features_as_of(
    prices: pl.DataFrame,
    *,
    as_of: date,
    feature_names: list[str],
    symbols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the feature matrix for each symbol's last bar at-or-before `as_of`.

    Returns (X, valid_symbols) where X has one row per valid symbol.
    Symbols missing the feature window (warmup not satisfied) are dropped.
    """
    feats = add_technical_features(prices)
    feats = feats.filter(pl.col("date") <= as_of).sort(["symbol", "date"])
    if symbols is not None:
        feats = feats.filter(pl.col("symbol").is_in(symbols))

    # Last row per symbol that has all features finite.
    last_rows = feats.group_by("symbol", maintain_order=True).tail(1).drop_nulls(subset=feature_names)
    finite_mask = pl.all_horizontal([pl.col(c).is_finite() for c in feature_names])
    last_rows = last_rows.filter(finite_mask)

    if last_rows.is_empty():
        return np.empty((0, len(feature_names)), dtype=np.float32), []

    valid = last_rows["symbol"].to_list()
    X = last_rows.select(feature_names).to_numpy().astype(np.float32)
    return X, valid


# ------------------------------------------------------------------
# Recommendation policy
# ------------------------------------------------------------------
def _confidence_band(score: float) -> str:
    a = abs(score)
    if a >= 0.30:
        return "high"
    if a >= 0.15:
        return "medium"
    return "low"


def recommend(
    bundle: ModelBundle,
    prices: pl.DataFrame,
    *,
    as_of: date,
    symbols: list[str] | None = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    explain: bool = False,
    top_k_drivers: int = 5,
) -> list[Recommendation]:
    """
    Build features → predict calibrated probs → emit per-symbol recommendation.

    When `explain=True`, each Recommendation gets a `top_drivers` list of
    the K features whose SHAP contributions to the BUY-minus-SELL score
    have the largest absolute magnitude.

    Symbols that fail the feature warmup at `as_of` are silently omitted.
    """
    X, valid_symbols = features_as_of(
        prices, as_of=as_of, feature_names=bundle.feature_names, symbols=symbols
    )
    if X.size == 0:
        return []

    cal = predict_calibrated(bundle, X)
    drivers_per_row: list[list[FeatureContribution]] | None = None
    if explain:
        contribs = shap_score_contributions(bundle, X)
        drivers_per_row = top_drivers(contribs, X.astype(np.float64), bundle.feature_names, k=top_k_drivers)

    out: list[Recommendation] = []
    for i, sym in enumerate(valid_symbols):
        p_neg = float(cal[i, _CLASS_NEG])
        p_zero = float(cal[i, 1])
        p_pos = float(cal[i, _CLASS_POS])
        score = p_pos - p_neg
        if score > threshold:
            action = "BUY"
        elif score < -threshold:
            action = "SELL"
        else:
            action = "HOLD"
        out.append(
            Recommendation(
                symbol=sym,
                as_of=as_of,
                prob_neg1=p_neg,
                prob_zero=p_zero,
                prob_pos1=p_pos,
                score=score,
                action=action,
                confidence=_confidence_band(score),
                top_drivers=drivers_per_row[i] if drivers_per_row is not None else None,
            )
        )
    return out


__all__ = [
    "DEFAULT_DECISION_THRESHOLD",
    "FeatureContribution",
    "ModelBundle",
    "Recommendation",
    "features_as_of",
    "load_bundle",
    "predict_calibrated",
    "recommend",
    "shap_score_contributions",
    "top_drivers",
]
