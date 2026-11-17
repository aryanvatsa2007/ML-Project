"""Run the whole project:  python run_pipeline.py   (downloads data, trains, evaluates, writes docs/results.json)

Options:
  --skip-download   reuse data/raw/*.csv
  --synthetic       fake data, ONLY to test that the code runs (results are labelled DEMO)
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src import config as C
from src import data, evaluate as ev, features as F, models as M, portfolio as P


def main(synthetic=False, skip_download=False):
    # 1) data ----------------------------------------------------------------
    if synthetic:
        close, volume = data.make_synthetic()
    else:
        close, volume = data.load() if skip_download else data.download()
    quality = data.quality_report(close, volume)
    close, volume = data.clean(close, volume)
    print("Data quality:", quality)

    # 2) features + chronological split --------------------------------------
    panel = F.build_panel(close, volume)
    train, val, test = F.time_split(panel)
    print(f"rows  train={len(train)}  val={len(val)}  test={len(test)}")

    # 3) tune on validation, refit on train+val, predict test once -----------
    best, search_log = M.tune(train, val)
    print("best params:", best)
    preds, fitted = M.predict_all(train, val, test, best)

    val_rmse = {k: M.rmse(val["y"], v[0]) for k, v in preds.items()}
    test_preds = {k: v[1] for k, v in preds.items()}
    table = ev.metrics_table(test["y"].values, test_preds)
    for row in table:
        row["val_rmse"] = val_rmse[row["model"]]
    best_name = min(["Ridge", "Random Forest"], key=lambda n: val_rmse[n])   # chosen on VALIDATION, not test
    print(pd.DataFrame(table).round(4).to_string(index=False))
    print("chosen model (by validation RMSE):", best_name)

    best_pred, persist_pred = test_preds[best_name], test_preds[ev.PERSIST]

    # 4) error analysis + explanation ----------------------------------------
    rf_imp, ridge_coef = ev.importances(fitted["Random Forest"], fitted["Ridge"], test)

    # 5) per-stock series for the website (every 3rd day keeps the JSON small)
    series = {}
    for t, g in test.assign(best=best_pred, persist=persist_pred).groupby("ticker"):
        g = g.iloc[::3]
        series[t] = {"dates": [str(d.date()) for d in g["date"]],
                     "actual": ev.ann_pct(g["y"]).round(2).tolist(),
                     "model": ev.ann_pct(g["best"]).round(2).tolist(),
                     "persistence": ev.ann_pct(g["persist"]).round(2).tolist()}

    # 6) portfolio level: does a better stock forecast give a better portfolio forecast?
    returns = F.log_returns(close)
    tt = test.assign(best=best_pred, persist=persist_pred)
    wide = lambda col: tt.pivot(index="date", columns="ticker", values=col)[C.TICKERS]
    pred_lv = {"ML model": wide("best"), "Persistence": wide("persist")}
    port_out = {}
    for pname, w in C.PORTFOLIOS.items():
        bt, summ = P.backtest(returns, pred_lv, w)
        bt = bt.iloc[::5]
        port_out[pname] = {
            "summary": summ,
            "dates": [str(d.date()) for d in bt.index],
            "realised": (bt["realised"] * np.sqrt(252) * 100).round(2).tolist(),
            "ml": (bt["ML model"] * np.sqrt(252) * 100).round(2).tolist(),
            "persistence": (bt["Persistence"] * np.sqrt(252) * 100).round(2).tolist()}
        print(pname, summ)

    # 7) "today's" snapshot: predict from the most recent feature row ----------
    latest = panel.sort_values("date").groupby("ticker").tail(1).set_index("ticker").loc[C.TICKERS]
    model_now = np.exp(fitted[best_name].predict(latest[F.FEATURES]))
    persist_now = np.exp(latest["lvol_21"].values)
    corr = np.corrcoef(returns[C.TICKERS].dropna().iloc[-C.CORR_WINDOW:].values.T)
    ann = np.sqrt(C.TRADING_DAYS) * 100

    result = {
        "meta": {
            "synthetic_demo": synthetic,
            "data_source": "SYNTHETIC demo data (code test only)" if synthetic else "Yahoo Finance via yfinance (adjusted close, volume)",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tickers": C.TICKERS, "horizon_days": C.HORIZON, "best_model": best_name,
            "split": {"train_end": C.TRAIN_END, "val_end": C.VAL_END,
                      "n_train": len(train), "n_val": len(val), "n_test": len(test)},
            "quality": quality, "features": F.FEATURES, "best_params": best},
        "metrics": table, "search_log": search_log,
        "per_ticker": ev.per_ticker(test, best_pred, persist_pred),
        "by_regime": ev.by_market_regime(test, best_pred, persist_pred),
        "worst_errors": ev.worst_errors(test, best_pred, persist_pred),
        "rf_importance": rf_imp, "ridge_coefficients": ridge_coef,
        "series": series, "portfolio_backtest": port_out,
        "current": {"as_of": str(latest["date"].max().date()), "tickers": C.TICKERS,
                    "model_vol": (model_now * ann).round(2).tolist(),
                    "persistence_vol": (persist_now * ann).round(2).tolist(),
                    "corr": np.round(corr, 3).tolist(),
                    "presets": {k: [round(v[t], 4) for t in C.TICKERS] for k, v in C.PORTFOLIOS.items()}},
    }
    C.DOCS_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.DOCS_DIR / "results.json").write_text(json.dumps(result))
    pd.DataFrame(table).to_csv(C.RESULTS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(search_log).to_csv(C.RESULTS_DIR / "tuning_log.csv", index=False)
    print("wrote docs/results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    main(a.synthetic, a.skip_download)
