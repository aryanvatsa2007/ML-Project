"""Baselines and models. Tuning uses the VALIDATION set only."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from . import config as C
from .features import FEATURES, HAR_FEATURES


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def make_ridge(alpha):
    # Scaler lives INSIDE the pipeline, so it is fitted on training rows only (no leakage)
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def make_rf(max_depth, min_samples_leaf):
    return RandomForestRegressor(n_estimators=200, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 n_jobs=-1, random_state=C.SEED)


def tune(train, val):
    """Pick hyper-parameters with the lowest validation RMSE. Returns best params + full search log."""
    log, best = [], {}
    ytr, yva = train["y"], val["y"]

    for a in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        m = make_ridge(a).fit(train[FEATURES], ytr)
        log.append({"model": "Ridge", "params": {"alpha": a}, "val_rmse": rmse(yva, m.predict(val[FEATURES]))})

    for d in [3, 5, 8]:
        for leaf in [20, 50, 100]:
            m = make_rf(d, leaf).fit(train[FEATURES], ytr)
            log.append({"model": "Random Forest", "params": {"max_depth": d, "min_samples_leaf": leaf},
                        "val_rmse": rmse(yva, m.predict(val[FEATURES]))})

    for name in ("Ridge", "Random Forest"):
        best[name] = min((l for l in log if l["model"] == name), key=lambda l: l["val_rmse"])["params"]
    return best, log


def build_model(name, params):
    if name == "Ridge":
        return make_ridge(**params)
    if name == "Random Forest":
        return make_rf(**params)
    raise ValueError(name)


def predict_all(train, val, test, best):
    """Returns {model_name: (val_pred, test_pred)} for 2 baselines, HAR, Ridge, RF.
    Validation predictions come from models fitted on train only.
    Test predictions come from models re-fitted on train+val (more recent data), then used ONCE."""
    out, trval = {}, None
    import pandas as pd
    trval = pd.concat([train, val], ignore_index=True)

    # Baseline 1: predict the training-set average log-volatility
    out["Historical mean"] = (np.full(len(val), train["y"].mean()), np.full(len(test), trval["y"].mean()))
    # Baseline 2: persistence - "next month's volatility = the last 21 days' volatility"
    out["Persistence (last 21d vol)"] = (val["lvol_21"].values, test["lvol_21"].values)
    # Baseline 3: HAR - linear regression on short/medium/long past volatility only
    har_v = LinearRegression().fit(train[HAR_FEATURES], train["y"]).predict(val[HAR_FEATURES])
    har_t = LinearRegression().fit(trval[HAR_FEATURES], trval["y"]).predict(test[HAR_FEATURES])
    out["HAR (3 vol lags, OLS)"] = (har_v, har_t)
    # ML models with all features
    fitted = {}
    for name in ("Ridge", "Random Forest"):
        mv = build_model(name, best[name]).fit(train[FEATURES], train["y"])
        mt = build_model(name, best[name]).fit(trval[FEATURES], trval["y"])
        out[name] = (mv.predict(val[FEATURES]), mt.predict(test[FEATURES]))
        fitted[name] = mt
    return out, fitted
