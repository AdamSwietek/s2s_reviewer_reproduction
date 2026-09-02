"""Model-form sensitivity analyses for the fire fragility curves."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .fragility import FIRES, fit_5pl, logistic_5pl


def logistic_4pl(x, p_min, p_max, k, hill):
    """Symmetric four-parameter logistic response."""
    x = np.maximum(np.asarray(x, float), 1e-300)
    with np.errstate(over="ignore", invalid="ignore"):
        return p_min + (p_max - p_min) / (1 + (k / x) ** hill)


def fit_4pl(x, y) -> dict[str, object]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    base = y[x < np.quantile(x, .1)].mean()
    top = y[x > np.quantile(x, .9)].mean()
    parameters, _ = curve_fit(
        logistic_4pl, x, y,
        p0=[np.clip(base, 1e-3, .4), np.clip(top, .1, .99),
            np.median(x), 1.],
        bounds=([0, 0, 1e-12, .1], [.5, 1., x.max() * 5, 20.]),
        maxfev=50_000,
    )
    return {
        "p_min": float(parameters[0]), "p_max": float(parameters[1]),
        "k": float(parameters[2]), "hill": float(parameters[3]),
        "asymmetry": 1., "f50": float(parameters[2]),
        "parameters": parameters,
    }


def _scores(y, probability) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    y = np.asarray(y, float)
    log_loss = float(-np.mean(y * np.log(probability)
                              + (1 - y) * np.log(1 - probability)))
    brier = float(np.mean((y - probability) ** 2))
    return log_loss, brier


def compare_4pl_5pl(data: pd.DataFrame, n_folds: int = 10,
                    seed: int = 27
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare model forms using held-out 250-m spatial grid cells."""
    exposed = data[data.exposed.eq(1)].copy()
    fitters = {"4PL": fit_4pl, "5PL": fit_5pl}
    evaluators = {"4PL": logistic_4pl, "5PL": logistic_5pl}
    full_rows, prediction_rows = [], []
    rng = np.random.default_rng(seed)

    for fire in FIRES:
        frame = exposed[exposed.fire.eq(fire)].copy().reset_index(drop=True)
        x = frame.F_destroyed_wmean.to_numpy(float)
        y = frame.is_destroyed.to_numpy(int)
        fitted = {}
        for model, fitter in fitters.items():
            fit = fitter(x, y)
            fitted[model] = fit
            probability = evaluators[model](x, *fit["parameters"])
            log_loss, brier = _scores(y, probability)
            full_rows.append({
                "fire": fire, "model": model, "n": len(frame),
                "p_min": fit["p_min"], "p_max": fit["p_max"],
                "k": fit["k"], "hill": fit["hill"],
                "asymmetry": fit["asymmetry"], "f50": fit["f50"],
                "in_sample_log_loss": log_loss,
                "in_sample_brier": brier,
            })

        cells = frame.grid_id.drop_duplicates().to_numpy()
        rng.shuffle(cells)
        fold_lookup = {cell: index % n_folds for index, cell in enumerate(cells)}
        folds = frame.grid_id.map(fold_lookup).to_numpy(int)
        for fold in range(n_folds):
            train, test = folds != fold, folds == fold
            for model, fitter in fitters.items():
                try:
                    fit = fitter(x[train], y[train])
                    probability = evaluators[model](x[test], *fit["parameters"])
                except (RuntimeError, ValueError):
                    probability = np.full(test.sum(), np.nan)
                for index, p in zip(np.flatnonzero(test), probability):
                    prediction_rows.append({
                        "fire": fire, "model": model, "fold": fold,
                        "row_id": index, "observed": y[index],
                        "predicted": p,
                    })

    full = pd.DataFrame(full_rows)
    predictions = pd.DataFrame(prediction_rows).dropna(subset=["predicted"])
    cv_rows = []
    for (fire, model), frame in predictions.groupby(["fire", "model"]):
        log_loss, brier = _scores(frame.observed, frame.predicted)
        cv_rows.append({
            "fire": fire, "model": model, "n_predicted": len(frame),
            "spatial_cv_log_loss": log_loss,
            "spatial_cv_brier": brier,
        })
    cv = pd.DataFrame(cv_rows)
    return full, cv, predictions
