"""Central settings. Change things here, not inside the other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"

# 10 large US stocks from different sectors + SPY (market index, used only as a feature source)
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "JNJ", "PG", "WMT", "KO"]
MARKET = "SPY"

START = "2015-01-01"
END = None                    # None = up to today

HORIZON = 21                  # predict volatility over the NEXT 21 trading days (~1 month)
TRAIN_END = "2021-12-31"      # train: start .. 2021
VAL_END = "2022-12-31"        # validation (tuning): 2022
                              # test (touched once at the end): 2023 onwards

TRADING_DAYS = 252
CORR_WINDOW = 126             # days used to estimate correlations for portfolio risk
VAR_Z = 1.645                 # 95% one-sided normal quantile
SEED = 42
EPS = 1e-8

PORTFOLIOS = {
    "Equal weight": {t: 1 / len(TICKERS) for t in TICKERS},
    "Tech heavy": {"AAPL": .2, "MSFT": .2, "NVDA": .2, "AMZN": .2, "JPM": .05,
                   "XOM": .05, "JNJ": .04, "PG": .02, "WMT": .02, "KO": .02},
}
