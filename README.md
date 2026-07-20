# Stock Regime Detection & Forecasting System

A production-grade quantitative research system that detects market regimes in NIFTY 50 stocks using multiple unsupervised learning approaches, trains regime-aware prediction models, and forecasts future price movements with full explainability via SHAP.

## Overview

Most stock prediction models treat all market conditions the same. This system first identifies **what type of market we're in** — Bull, Bear, Sideways, or High Volatility — then uses regime-specific models to make more accurate predictions. A Bear market model learns Bear patterns. A Bull market model learns Bull patterns.

## Architecture

```
Raw OHLCV Data (yfinance)
        ↓
Feature Engineering (40+ technical indicators)
        ↓
Regime Detection (KMeans + GMM + HMM compared)
        ↓
Regime-Aware LightGBM (one model per regime)
        ↓
Walk-Forward Backtesting (TimeSeriesSplit, gap=60)
        ↓
SHAP Explainability + Prophet Forecasting
        ↓
MongoDB Storage + Streamlit Dashboard
```

## Tech Stack

| Category | Tools |
|---|---|
| Data | yfinance, pandas |
| Features | ta, numpy |
| Regime Detection | statsmodels (HMM), sklearn (KMeans, GMM) |
| Prediction | LightGBM |
| Explainability | SHAP |
| Forecasting | Prophet |
| Database | MongoDB, pymongo |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

## Features

**Regime Detection:**
- Hidden Markov Model via Markov Switching (statsmodels)
- KMeans clustering baseline
- Gaussian Mixture Model with soft probability assignments
- Comparison of all three approaches

**Technical Indicators (40+):**
- Momentum: RSI, Stochastic RSI, Williams %R
- Trend: MACD, CCI, Rolling Close Ratios
- Volatility: Bollinger Bands, ATR, Volatility_5/20
- Volume: OBV, Volume Ratio, Volume Change
- Lag features: RSI, MACD, ATR lagged 1/2/3 days
- Regime features: Returns_5, High_Low_Range, Gap

**Prediction:**
- LightGBM trained separately per regime
- 5-day forward return prediction
- TimeSeriesSplit with gap=60 to prevent leakage
- Precision, Sharpe Ratio, Max Drawdown evaluation

**Explainability:**
- SHAP values per prediction
- Feature importance per regime
- Which indicators drove each signal

**Forecasting:**
- Prophet price forecasting for next 5/10/30 days
- Confidence intervals shown on dashboard

## Project Structure

```
stock_regime/
├── data.py          # yfinance download + MongoDB caching
├── database.py      # MongoDB operations (StockDB class)
├── features.py      # technical indicator engineering
├── regime.py        # HMM, KMeans, GMM regime detection
├── model.py         # regime-aware LightGBM training + backtest
├── evaluate.py      # metrics + SHAP analysis
├── forecast.py      # Prophet price forecasting
├── train.py         # offline training pipeline
├── app.py           # Streamlit dashboard
└── tests/
    └── test_database.py
```

## Installation

```bash
git clone https://github.com/m4nn2609-dot/stock-regime-detection.git
cd stock-regime-detection
pip install -r requirements.txt
```

**Requirements:**
```
yfinance
ta
pandas
numpy
scikit-learn
lightgbm
shap
prophet
statsmodels
pymongo
streamlit
plotly
pytest
```

## Usage

**Step 1 — Train all stocks (run once):**
```bash
python train.py
```

**Step 2 — Launch dashboard:**
```bash
streamlit run app.py
```

## Results

| Metric | Value |
|---|---|
| Prediction Horizon | 5 days |
| Backtesting Method | TimeSeriesSplit (n=5, gap=60) |
| Number of Regimes | 4 |
| Stocks Covered | NIFTY 50 |

## How It Differs From Standard Prediction

| Standard Approach | This System |
|---|---|
| One model for all conditions | Separate model per regime |
| Predict UP/DOWN | Predict UP/DOWN within regime context |
| Black box predictions | SHAP explainability per signal |
| No market context | Regime label + transition awareness |

## Related Projects

- [Stock Price Prediction V1](https://github.com/m4nn2609-dot/stock-prediction) — NIFTY 50 ensemble (RF + XGBoost + LSTM)
- [Stock Price Prediction V2](https://github.com/m4nn2609-dot/Reets-Project) — S&P 500 with purged cross-validation

