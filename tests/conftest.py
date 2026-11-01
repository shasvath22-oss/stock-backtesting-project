"""Deterministic vendor-shaped fixtures, used only inside automated tests."""
import pandas as pd
import pytest


@pytest.fixture
def fake_yfinance(monkeypatch):
    from trend_analyzer.data import sessions,synthetic_bars
    import yfinance as yf
    class Ticker:
        def __init__(self,ticker):
            self.ticker = ticker
        def history(self,**kwargs):
            dates = sessions(kwargs["start"],pd.Timestamp(kwargs["end"])-pd.Timedelta(days=1))
            rows = synthetic_bars([self.ticker],dates,42).set_index("date")
            rows.index.name = "Date"
            return rows.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close",
                "adj_close":"Adj Close","volume":"Volume","dividend":"Dividends","split":"Stock Splits"})
    monkeypatch.setattr(yf,"Ticker",Ticker)
    monkeypatch.setattr(yf,"set_tz_cache_location",lambda path:None)
