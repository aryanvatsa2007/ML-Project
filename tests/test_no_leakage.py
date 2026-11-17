"""Sanity tests for the most important ML rule: the model must never see the future."""
import numpy as np
import pandas as pd
from src import config as C, data, features as F


def _panel():
    close, volume = data.make_synthetic(end="2019-12-31")
    return close, volume, F.build_panel(close, volume)


def test_features_do_not_change_when_future_prices_change():
    close, volume, panel = _panel()
    cut = close.index[800]
    altered = close.copy()
    altered.loc[altered.index > cut] *= 1.5          # wreck the future
    panel2 = F.build_panel(altered, volume)
    a = panel[panel["date"] <= cut].reset_index(drop=True)
    b = panel2[panel2["date"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[F.FEATURES], b[F.FEATURES])   # features identical up to the cut


def test_target_uses_only_the_next_horizon_days():
    close, volume, panel = _panel()
    r = F.log_returns(close)["AAPL"]
    row = panel[(panel["ticker"] == "AAPL")].iloc[100]
    i = r.index.get_loc(row["date"])
    manual = np.log(r.iloc[i + 1:i + 1 + C.HORIZON].std() + C.EPS)
    assert abs(manual - row["y"]) < 1e-9


def test_split_is_chronological_with_gap():
    close, volume = data.make_synthetic()            # full date range so val/test periods exist
    panel = F.build_panel(close, volume)
    train, val, test = F.time_split(panel)
    assert train["date"].max() < val["date"].min() < val["date"].max() < test["date"].min()
    gap_days = len(pd.bdate_range(train["date"].max(), val["date"].min())) - 2
    assert gap_days >= C.HORIZON - 3                 # no overlapping target windows between train and val
