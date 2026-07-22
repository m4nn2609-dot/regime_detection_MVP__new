import pytest


from database import StockDB

from data import stock_data

import yfinance as yf
import numpy as np
import pandas as pd
from model import train_regime_model
from shap_analysis import compute_shap_for_regime_models

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
    result = db.load_results("FAKE_TICKER", "kmeans_labels")
    assert result is None

def test_load_data():
    sd = stock_data()
    stocks = sd.load_data()
    assert len(stocks) > 0
    assert "RELIANCE.NS" in stocks.keys()
    assert not stocks["RELIANCE.NS"].empty
    assert "Close" in stocks["RELIANCE.NS"].columns

@pytest.fixture(scope="module")
def synthetic_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "Feature_1": np.random.randn(n),
        "Feature_2": np.random.randn(n),
        "Feature_3": np.random.randn(n),
        "Target": np.random.randint(0, 2, n),
         "kmeans_labels": np.random.randint(0, 3, n),
    })
    return df

@pytest.fixture(scope="module")
def predictors():
    return ["Feature_1", "Feature_2", "Feature_3"]

@pytest.fixture(scope="module")
def regime_models(synthetic_df, predictors):
    return train_regime_model(synthetic_df, predictors, regime_col="kmeans_labels")


def test_train_regime_model_returns_models(regime_models):
    assert len(regime_models) > 0
    for regime_id, model in regime_models.items():
        assert hasattr(model, "predict_proba")


def test_compute_shap_returns_dataframe(regime_models, synthetic_df, predictors):
    shap_df = compute_shap_for_regime_models(regime_models, synthetic_df, predictors, "kmeans_labels")
    assert not shap_df.empty
    for col in predictors:
        assert col in shap_df.columns
    assert "Regime" in shap_df.columns

def test_compute_shap_row_count_matches_regime_rows(regime_models, synthetic_df, predictors):
    shap_df = compute_shap_for_regime_models(regime_models, synthetic_df, predictors, "kmeans_labels")
    expected_rows = synthetic_df[synthetic_df["kmeans_labels"].isin(regime_models.keys())].shape[0]
    assert len(shap_df) == expected_rows


def test_compute_shap_empty_models_returns_empty_df(synthetic_df, predictors):
    shap_df = compute_shap_for_regime_models({}, synthetic_df, predictors, "kmeans_labels")
    assert shap_df.empty


def test_save_and_load_shap():
    db = StockDB()
    df = pd.DataFrame({
        "Feature_1": [0.1, 0.2, 0.3],
        "Feature_2": [-0.1, -0.2, -0.3],
        "Regime": [0, 0, 1],
    }, index=pd.date_range("2024-01-01", periods=3))
    df.index.name = "Date"


    db.save_shap(df.copy(), "TEST_TICKER", "kmeans_labels")
    loaded = db.load_shap("TEST_TICKER", "kmeans_labels")

    assert not loaded.empty
    assert "Feature_1" in loaded.columns
    assert "Regime" in loaded.columns
    assert len(loaded) == 3


def test_load_shap_missing_ticker_returns_empty():
    db = StockDB()
    loaded = db.load_shap("FAKE_TICKER_NOT_PRESENT", "kmeans_labels")
    assert loaded.empty
