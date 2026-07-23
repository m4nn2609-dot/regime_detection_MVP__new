import pandas as pd
import numpy as np
import traceback
from features import add_features,create_target,create_forward,regime_features
from database import StockDB
from data import stock_data
from evaluate import evaluate
from shap_analysis import compute_shap_for_regime_models
from model import train_regime_model, backtest_regime,label_regimes

from forecast import forecast_price

from regime import fit_kmeans,fit_gmm,fit_hmm,forecast_regime
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
db=StockDB()
db.clear_all()
stocks=stock_data().load_data()
print(stocks.keys())
#stocks = {"RELIANCE.NS": stock_data().load_data()["RELIANCE.NS"]}
for ticker,df in stocks.items():
    if df.empty:
        print(f"{ticker} not available due to library outsourcing issue.")
        continue
    try:
        print(df.shape)
        print(df.dtypes)
        print(df.head())
        df=add_features(df)
        print("add_features done")
        df = create_forward(df)
        print("create_forward done")
        df=create_target(df)
        print("create_target done")

        df=regime_features(df)
        print("regime_features done")

        df = df.replace([np.inf, -np.inf], np.nan)
        df=df.dropna()
        print(f"{ticker}:{len(df)} rows after dropna")
        if df.empty:
            print(f"{ticker}: no rows left after dropna (likely insufficient history for rolling features), skipping.")
            continue


        df = fit_kmeans(df)
        df = fit_gmm(df)
        try:
            df = fit_hmm(df)
            methods = ["kmeans_labels", "gmm_labels", "hmm_labels"]
        except Exception as e:
            print(f"{ticker}: fit_hmm failed ({e}), continuing without HMM regimes.")
            traceback.print_exc()
            df["hmm_labels"] = -1  # placeholder so downstream column selection doesn't break
            methods = ["kmeans_labels", "gmm_labels"]
        for method in methods:
            df = label_regimes(df, method)

        name_cols = [f"{m}_name" for m in methods]
        if "hmm_labels_name" in df.columns and "hmm_labels_name" not in name_cols:
            name_cols.append("hmm_labels_name")

        feature_cols = [c for c in df.columns if c not in
                        ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target',
                         'Forward'] + name_cols
                        and not c.endswith('_labels') and c != 'Date']
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=feature_cols)
        db.save_regime(df[['kmeans_labels', 'gmm_labels', 'hmm_labels'] + name_cols], ticker)
        print(df.columns.tolist())

        predictors = df.drop(
            ['Stock Splits', 'Dividends', 'Close', 'Open', 'Target', 'Forward', 'Volume', 'High', 'Low',
             'kmeans_labels', 'gmm_labels', 'hmm_labels']+name_cols, axis=1).columns.tolist()

        for regime_method in methods:
            try:
                preds = backtest_regime(df, predictors, regime_method)
                models = train_regime_model(df, predictors, regime_method)
                shap_df = compute_shap_for_regime_models(models, df, predictors, regime_method)

                if not shap_df.empty:
                    db.save_shap(shap_df, ticker, regime_method)
                else:
                    print(f"{ticker}/{regime_method}: SHAP empty, skipping save.")

                if preds is not None and not preds.empty and preds['Target'].nunique() > 1:
                    res = evaluate(preds, df)
                    db.save_predictions(preds, ticker)
                    db.save_results(ticker, res['precision'], res['sharpe'], res['max_dd'], regime_method=regime_method,
                                    strategy_return=res['strategy_return'], buy_hold_return=res['buy_hold_return'],
                                    excess_return=res['excess_return'])
                else:
                    print(f"{ticker}/{regime_method}: not enough backtest predictions to evaluate, skipping.")
            except Exception as e:
                print(f"{ticker}/{regime_method} pipeline step failed: {e}")
                traceback.print_exc()
                continue


        try:
            forecast = forecast_price(df)
            for method in methods:
                regime_forecast = forecast_regime(df, forecast, method)
                forecast = forecast.join(regime_forecast, how="left")
            db.save_forecast(forecast, ticker)
        except Exception as e:
            print(f"{ticker}: forecast_price failed: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"{ticker} went through a problem:{e}.")
        traceback.print_exc()