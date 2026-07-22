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
    name_col = f"{regime_method}_name"

    if price_df.empty:
        st.warning("No price data found for this ticker yet. Run train.py first.")
    elif not regime_df.empty and name_col in regime_df.columns:
        merged = price_df[["Close"]].join(regime_df[[name_col]], how="inner")
        merged = merged.rename(columns={name_col: "Regime"})

        color_map = {
            "Bull": "#3DDC97", "Bear": "#FF6B6B",
            "Sideways": "#9AA5B1", "High Volatility": "#FFC857",
            "Unknown": "#555555",
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged.index, y=merged["Close"], mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            showlegend=False, hoverinfo="skip"
        ))
        for regime_name, group in merged.groupby("Regime"):
            fig.add_trace(go.Scatter(
                x=group.index, y=group["Close"], mode="markers",
                name=regime_name,
                marker=dict(size=4, color=color_map.get(regime_name, "#888888")),
            ))
        fig.update_layout(template="plotly_dark", legend_title_text="Regime")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df["Close"], mode="lines", name="Close"))
        st.plotly_chart(fig, use_container_width=True)
        st.info("No regime label data found yet for this method — rerun train.py.")

with tab2:
    st.subheader(f"{ticker} — Backtest Results ({regime_method})")
    result = db.load_results(ticker, regime_method)
    if result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{result.get('precision', 0):.3f}")
        col2.metric("Sharpe Ratio", f"{result.get('sharpe', 0):.3f}")
        col3.metric("Max Drawdown", f"{result.get('max_dd', 0):.3%}")

        col4, col5 = st.columns(2)
        excess = result.get('excess_return') or 0
        col4.metric("Strategy Return", f"{result.get('strategy_return', 0):.2%}")
        col5.metric("Excess vs Buy & Hold", f"{excess:.2%}", delta=f"{excess:.2%}")
    else:
        st.warning("No results found for this ticker/method yet.")

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
    price_df = db.load_stock_data(ticker)
    if forecast_df.empty:
        st.warning("No forecast data found for this ticker yet.")
    else:
        cutoff = forecast_df.index.max() - pd.Timedelta(days=210)
        recent_forecast = forecast_df[forecast_df.index >= cutoff]
        recent_price = price_df[price_df.index >= cutoff] if not price_df.empty else price_df

        fig5 = go.Figure()
        if not recent_price.empty:
            fig5.add_trace(go.Scatter(x=recent_price.index, y=recent_price["Close"], mode="lines",
                                       name="Actual", line=dict(color="#4C9AFF", width=2)))

        if "yhat_lower" in recent_forecast.columns and "yhat_upper" in recent_forecast.columns:
            fig5.add_trace(go.Scatter(x=recent_forecast.index, y=recent_forecast["yhat_upper"],
                                       mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig5.add_trace(go.Scatter(x=recent_forecast.index, y=recent_forecast["yhat_lower"],
                                       mode="lines", line=dict(width=0), fill="tonexty",
                                       fillcolor="rgba(255,140,140,0.2)", name="Confidence Interval", hoverinfo="skip"))

        fig5.add_trace(go.Scatter(x=recent_forecast.index, y=recent_forecast["yhat"], mode="lines",
                                   name="Forecast", line=dict(color="#FF8C8C", width=2, dash="dot")))

        fig5.update_layout(template="plotly_dark", hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

        forecast_regime_col = f"{regime_method}_forecast_name"
        if forecast_regime_col in recent_forecast.columns:
            upcoming = recent_forecast[recent_forecast.index > price_df.index.max()][[forecast_regime_col]].dropna()
            if not upcoming.empty:
                st.caption("Forecasted regime over the next few days:")
                st.dataframe(upcoming.rename(columns={forecast_regime_col: "Forecast Regime"}))