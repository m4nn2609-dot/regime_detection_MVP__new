import pytest

from database import StockDB

from data import stock_data

import yfinance as yf

@pytest.fixture(scope="session", autouse=True)

def setup_data():
    db = StockDB()
    df = yf.Ticker("RELIANCE.NS").history(period="1y")
    db.save_stock_data(df, "RELIANCE.NS")

def test_save_and_load_stock():
    db = StockDB()
    sd = stock_data()
    stocks = sd.load_data()
    assert len(stocks) > 0

def test_load_stock_data():
    db = StockDB()
    df = db.load_stock_data("RELIANCE.NS")
    assert not df.empty
    assert "Close" in df.columns

def test_load_results_empty():
    db = StockDB()
    result = db.load_results("FAKE_TICKER")
    assert result is None

def test_load_data():
    sd = stock_data()
    stocks = sd.load_data()
    assert len(stocks) > 0
    assert "RELIANCE.NS" in stocks.keys()
    assert not stocks["RELIANCE.NS"].empty
    assert "Close" in stocks["RELIANCE.NS"].columns
    