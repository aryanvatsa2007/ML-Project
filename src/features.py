"""Turn prices into a (date, ticker) table of features X and target y.

Every feature at date t uses ONLY data up to and including day t.
The target at date t is the log of the realised volatility over days t+1 .. t+HORIZON.
"""
import numpy as np
import pandas as pd
from . import config as C

FEATURES = ["lvol_5", "lvol_21", "lvol_63", "ret_5", "ret_21", "drawdown_63",
            "vol_ratio", "volume_ratio", "abs_ret_1", "mkt_lvol_21", "mkt_ret_21"]
HAR_FEATURES = ["lvol_5", "lvol_21", "lvol_63"]   # short / medium / long past volatility


def log_returns(close):
    return np.log(close).diff()


def build_panel(close, volume, horizon=C.HORIZON):
    r = log_returns(close)
    m = r[C.MARKET]
    mkt_lvol = np.log(m.rolling(21).std() + C.EPS)
    mkt_ret = m.rolling(21).sum()
    parts = []
    for t in C.TICKERS:
        x = r[t]
        v5, v21, v63 = (x.rolling(k).std() for k in (5, 21, 63))
        vol = volume[t].replace(0, np.nan)
        d = pd.DataFrame({
            "lvol_5": np.log(v5 + C.EPS),
            "lvol_21": np.log(v21 + C.EPS),
            "lvol_63": np.log(v63 + C.EPS),
            "ret_5": x.rolling(5).sum(),
            "ret_21": x.rolling(21).sum(),
            "drawdown_63": close[t] / close[t].rolling(63).max() - 1,
            "vol_ratio": np.log((v5 + C.EPS) / (v63 + C.EPS)),
            "volume_ratio": np.log(vol / vol.rolling(63).mean()),
            "abs_ret_1": x.abs(),
            "mkt_lvol_21": mkt_lvol,
            "mkt_ret_21": mkt_ret,
            # target: std of returns on days t+1..t+horizon (shift(-h) moves the future back to t)
            "y": np.log(x.rolling(horizon).std().shift(-horizon) + C.EPS),
        })
        d["ticker"] = t
        parts.append(d)
    panel = pd.concat(parts)
    panel.index.name = "date"
    panel = panel.reset_index()
    return panel.dropna(subset=FEATURES).reset_index(drop=True)   # y may stay NaN for the latest dates


def time_split(panel, horizon=C.HORIZON):
    """Chronological split. We drop `horizon` dates at the end of train and validation because
    their targets look into the next period (overlapping windows would leak)."""
    dates = np.sort(panel["date"].unique())
    tr_all = dates[dates <= np.datetime64(C.TRAIN_END)]
    va_all = dates[(dates > np.datetime64(C.TRAIN_END)) & (dates <= np.datetime64(C.VAL_END))]
    te_all = dates[dates > np.datetime64(C.VAL_END)]
    train_d, val_d = tr_all[:-horizon], va_all[:-horizon]
    test_d = te_all[:-horizon]            # the last `horizon` dates have no known target yet
    pick = lambda ds: panel[panel["date"].isin(ds) & panel["y"].notna()].reset_index(drop=True)
    return pick(train_d), pick(val_d), pick(test_d)
