import pandas as pd

from features import add_features,create_target,create_forward,regime_features
from database import StockDB
from data import stock_data
from evaluate import evaluate

from forecast import forecast_price
from model import backtest_regime
from regime import fit_kmeans,fit_gmm,fit_hmm
db=StockDB()
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

        df=df.replace([float('inf'), float('-inf')], pd.NA)
        df=df.dropna()
        print(f"{ticker}:{len(df)} rows after dropna")

        df=fit_kmeans(df)
        df=fit_gmm(df)
        df=fit_hmm(df)

        db.save_regime(df[['kmeans_labels','gmm_labels','hmm_labels']],ticker)
        print(df.columns.tolist())
        predictors=df.drop(['Stock Splits','Dividends','Close','Open','Target','Volume','High','Low'],axis=1).columns.tolist()

        kmeans_preds=(backtest_regime(df,predictors,"kmeans_label"))
        gmm_preds=(backtest_regime(df,predictors,"gmm_label"))
        hmm_preds=(backtest_regime(df,predictors,"hmm_label"))

        forecast=forecast_price(df)
        db.save_forecast(forecast,ticker)

        kmeans_res=evaluate(kmeans_preds,df)
        gmm_res=evaluate(gmm_preds,df)
        hmm_res=evaluate(hmm_preds,df)

        db.save_predictions(gmm_preds,ticker)
        db.save_predictions(kmeans_preds,ticker)
        db.save_predictions(hmm_preds,ticker)

        db.save_results(ticker,kmeans_res['precision'],kmeans_res['sharpe'],kmeans_res['max_dd'],regime_method='kmeans_labels')
        db.save_results(ticker,gmm_res['precision'],gmm_res['sharpe'],gmm_res['max_dd'],regime_method="gmm_labels")
        db.save_results(ticker,hmm_res['precision'],hmm_res['sharpe'],hmm_res['max_dd'],regime_method="hmm_labels")


    except Exception as e:
        print(f"{ticker} went through a problem:{e}.")
