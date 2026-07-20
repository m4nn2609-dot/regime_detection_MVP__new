import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import StockDB
from data import NIFTY50_TICKERS

st.set_page_config(page_title="Regime Detection Dashboard", layout="wide")

db = StockDB()

st.title("NIFTY 50 Regime Detection Dashboard")

ticker = st.sidebar.selectbox("Select Ticker", NIFTY50_TICKERS)
regime_method = st.sidebar.selectbox("Regime Method", ["kmeans_labels", "gmm_labels", "hmm_labels"])

tab1, tab2, tab3, tab4 = st.tabs(["Price & Regimes", "Model Performance", "SHAP Explanations", "Forecast"])

with tab1:
    st.subheader(f"{ticker} — Price History & Regimes")
    price_df = db.load_stock_data(ticker)
    regime_df = db.load_regimes(ticker)

    if price_df.empty:
        st.warning("No price data found for this ticker yet. Run train.py first.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Close"], mode="lines", name="Close"))
        st.plotly_chart(fig, use_container_width=True)

    if not regime_df.empty and regime_method in regime_df.columns:
        st.subheader("Regime Labels Over Time")
        fig2 = px.scatter(regime_df, x=regime_df.index, y=regime_method, color=regime_method)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No regime data found for this ticker/method yet.")

with tab2:
    st.subheader(f"{ticker} — Backtest Results ({regime_method})")
    result = db.load_results(ticker, regime_method)
    if result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{result.get('precision', 0):.3f}")
        col2.metric("Sharpe Ratio", f"{result.get('sharpe', 0):.3f}")
        col3.metric("Max Drawdown", f"{result.get('max_dd', 0):.3%}")
    else:
        st.warning("No results found for this ticker/method yet.")

    preds_df = db.load_predictions(ticker)
    if not preds_df.empty:
        st.subheader("Predictions vs Target")
        st.dataframe(preds_df.tail(50))

with tab3:
    st.subheader(f"{ticker} — SHAP Feature Importance ({regime_method})")
    shap_df = db.load_shap(ticker, regime_method)
    if shap_df.empty:
        st.warning("No SHAP data found for this ticker/method yet.")
    else:
        feature_cols = [c for c in shap_df.columns if c not in ["Date", "Regime"]]
        mean_abs_shap = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
        top_features = mean_abs_shap.head(15)

        fig3 = px.bar(
            x=top_features.values,
            y=top_features.index,
            orientation="h",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
            title="Global Feature Importance",
        )
        fig3.update_yaxes(autorange="reversed")
        st.plotly_chart(fig3, use_container_width=True)

        if "Regime" in shap_df.columns:
            st.subheader("Feature Importance by Regime")
            regime_choice = st.selectbox("Select Regime", sorted(shap_df["Regime"].unique()))
            subset = shap_df[shap_df["Regime"] == regime_choice][feature_cols]
            mean_abs_regime = subset.abs().mean().sort_values(ascending=False).head(15)
            fig4 = px.bar(
                x=mean_abs_regime.values,
                y=mean_abs_regime.index,
                orientation="h",
                labels={"x": "Mean |SHAP value|", "y": "Feature"},
            )
            fig4.update_yaxes(autorange="reversed")
            st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader(f"{ticker} — Price Forecast")
    forecast_df = db.load_forecast(ticker)
    if forecast_df.empty:
        st.warning("No forecast data found for this ticker yet.")
    else:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["yhat"], mode="lines", name="Forecast"))
        if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
            fig5.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
            fig5.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["yhat_lower"], mode="lines", line=dict(width=0), fill="tonexty", name="Confidence Interval"))
        st.plotly_chart(fig5, use_container_width=True)