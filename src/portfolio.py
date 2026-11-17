"""Turn per-stock volatility forecasts into portfolio risk."""
import numpy as np
import pandas as pd
from . import config as C


def portfolio_vol(w, vols, corr):
    """sigma_p = sqrt(w' S w), with S = D C D (D = diag of stock vols, C = correlation matrix)."""
    cov = np.outer(vols, vols) * corr
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(w, vols, corr):
    cov = np.outer(vols, vols) * corr
    sp = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / sp ** 2          # shares that sum to 1


def backtest(returns, pred_logvol, weights, horizon=C.HORIZON):
    """For each test date: build the portfolio vol forecast from {model stock vols + trailing correlations},
    and compare with the portfolio's REALISED vol over the next `horizon` days.
    pred_logvol: dict name -> DataFrame (date x ticker) of predicted log daily vol."""
    R = returns[C.TICKERS]
    w = np.array([weights[t] for t in C.TICKERS])
    port = R @ w
    realised = port.rolling(horizon).std().shift(-horizon)
    next_ret = port.shift(-1)
    dates = [d for d in next(iter(pred_logvol.values())).index if not np.isnan(realised.get(d, np.nan))]
    rows = []
    for d in dates:
        i = R.index.get_loc(d)
        corr = np.corrcoef(R.iloc[i - C.CORR_WINDOW + 1:i + 1].values.T)
        row = {"date": d, "realised": realised[d], "next_ret": next_ret[d]}
        for name, df in pred_logvol.items():
            row[name] = portfolio_vol(w, np.exp(df.loc[d].values), corr)
        rows.append(row)
    bt = pd.DataFrame(rows).set_index("date")
    summary = {}
    for name in pred_logvol:
        err = np.log(bt[name]) - np.log(bt["realised"])
        exceed = (bt["next_ret"] < -C.VAR_Z * bt[name]).mean()   # expect about 5%
        summary[name] = {"rmse_log": float(np.sqrt((err ** 2).mean())),
                         "var95_exceedance": float(exceed)}
    return bt, summary
