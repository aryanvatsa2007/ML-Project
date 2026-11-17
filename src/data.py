"""Getting the data: download from Yahoo Finance, load, quality-check.
A synthetic generator exists ONLY for testing the code offline (never for reported results)."""
import numpy as np
import pandas as pd
from . import config as C


def download(start=C.START, end=C.END):
    import yfinance as yf  # imported here so offline/test use does not need it
    names = C.TICKERS + [C.MARKET]
    df = yf.download(names, start=start, end=end, auto_adjust=True, progress=False)
    close, volume = df["Close"][names], df["Volume"][names]
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    close.to_csv(C.RAW_DIR / "close.csv")
    volume.to_csv(C.RAW_DIR / "volume.csv")
    return close, volume


def load():
    close = pd.read_csv(C.RAW_DIR / "close.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(C.RAW_DIR / "volume.csv", index_col=0, parse_dates=True)
    return close, volume


def quality_report(close, volume):
    """Simple checks we explain in the report: missing values, bad prices, extreme moves."""
    r = np.log(close).diff()
    return {
        "rows": int(len(close)),
        "first_date": str(close.index.min().date()),
        "last_date": str(close.index.max().date()),
        "missing_prices": int(close.isna().sum().sum()),
        "non_positive_prices": int((close <= 0).sum().sum()),
        "duplicate_dates": int(close.index.duplicated().sum()),
        "zero_volume_days": int((volume == 0).sum().sum()),
        "daily_moves_over_20pct": int((r.abs() > 0.20).sum().sum()),
    }


def clean(close, volume):
    """Forward-fill tiny gaps (<=3 days), then drop any date that is still incomplete."""
    close = close.sort_index().ffill(limit=3)
    volume = volume.sort_index().ffill(limit=3)
    ok = close.notna().all(axis=1) & volume.notna().all(axis=1)
    return close[ok], volume[ok]


def make_synthetic(start="2015-01-01", end="2026-10-30", seed=C.SEED):
    """Fake prices with volatility clustering (GARCH-like). For code testing only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    def garch(omega, alpha, beta):
        h, e, z = np.empty(n), np.empty(n), rng.standard_normal(n)
        h[0] = omega / (1 - alpha - beta)
        for t in range(n):
            if t:
                h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
            e[t] = np.sqrt(h[t]) * z[t]
        return e

    mkt = garch(2e-6, 0.10, 0.88)
    rets = {C.MARKET: mkt}
    for t in C.TICKERS:
        rets[t] = rng.uniform(0.6, 1.4) * mkt + garch(3e-6, 0.08, 0.89) + 0.0003
    rets = pd.DataFrame(rets, index=dates)
    close = 100 * np.exp(rets.cumsum())
    volume = pd.DataFrame(
        rng.lognormal(15, 0.3, rets.shape) * (1 + 40 * rets.abs().values), index=dates, columns=rets.columns)
    return close, volume
