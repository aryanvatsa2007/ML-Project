"""Metrics and error analysis (all on test data, computed once)."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from . import config as C
from .features import FEATURES
from .models import rmse

PERSIST = "Persistence (last 21d vol)"


def ann_pct(log_daily_vol):
    """log daily vol -> annualised volatility in %"""
    return np.exp(log_daily_vol) * np.sqrt(C.TRADING_DAYS) * 100


def metrics_table(y, preds):
    base = rmse(y, preds[PERSIST])
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "rmse": rmse(y, p), "mae": float(mean_absolute_error(y, p)),
                     "r2": float(r2_score(y, p)),
                     # >0 means better than persistence; <0 worse
                     "skill_vs_persistence": float(1 - rmse(y, p) ** 2 / base ** 2)})
    return rows


def per_ticker(test, best_pred, persist_pred):
    d = test.assign(best=best_pred, persist=persist_pred)
    rows = []
    for t, g in d.groupby("ticker"):
        rows.append({"ticker": t, "rmse_model": rmse(g["y"], g["best"]), "rmse_persistence": rmse(g["y"], g["persist"])})
    return rows


def by_market_regime(test, best_pred, persist_pred):
    d = test.assign(best=best_pred, persist=persist_pred)
    d["regime"] = pd.qcut(d["mkt_lvol_21"], 3, labels=["calm market", "normal market", "turbulent market"])
    return [{"regime": str(k), "rmse_model": rmse(g["y"], g["best"]), "rmse_persistence": rmse(g["y"], g["persist"]),
             "n": int(len(g))} for k, g in d.groupby("regime", observed=True)]


def worst_errors(test, best_pred, persist_pred, k=8):
    d = test.assign(best=best_pred, persist=persist_pred)
    d["abs_err"] = (d["best"] - d["y"]).abs()
    out = []
    for _, r in d.nlargest(k, "abs_err").iterrows():
        out.append({"date": str(r["date"].date()), "ticker": r["ticker"], "actual": float(ann_pct(r["y"])),
                    "model": float(ann_pct(r["best"])), "persistence": float(ann_pct(r["persist"]))})
    return out


def importances(rf_model, ridge_model, test):
    pi = permutation_importance(rf_model, test[FEATURES], test["y"], n_repeats=5, random_state=C.SEED,
                                scoring="neg_mean_squared_error")
    rf = sorted(({"feature": f, "importance": float(v)} for f, v in zip(FEATURES, pi.importances_mean)),
                key=lambda r: -r["importance"])
    coefs = ridge_model.named_steps["model"].coef_
    rg = sorted(({"feature": f, "coef": float(c)} for f, c in zip(FEATURES, coefs)), key=lambda r: -abs(r["coef"]))
    return rf, rg
