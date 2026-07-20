from pymongo import MongoClient
import pandas as pd
class StockDB():
    def __init__(self):
        self.client=MongoClient(host=r"mongodb://localhost:27017")
        self.db=self.client["Stocks"]
        self.stocks=self.db['Stock_Vals']
        self.results=self.db['Results']
        self.regimes=self.db['Regimes']
        self.SHAP=self.db['SHAP']
        self.predictions=self.db['Predictions']
        self.forecast=self.db['Forecast']
    def save_stock_data(self,df,val):

        df.reset_index(inplace=True)
        data=df.to_dict(orient='records')

        for rec in data:
            rec["Ticker"]=val
        self.stocks.insert_many(data)

    def load_all_stocks(self):
        df = pd.DataFrame(list(self.stocks.find()))
        if df.empty:
            return df
        df.drop(["_id"], axis=1, inplace=True)
        return df

    def load_stock_data(self,val):
        df=pd.DataFrame(list(self.stocks.find({"Ticker":val})))
        if df.empty:
            return df
        df.drop(["_id","Ticker"],axis=1,inplace=True)
        df.set_index("Date",inplace=True)
        return df
    def save_results(self,ticker,precision,sharpe,max_dd,regime_method):
        self.results.insert_one(
            {
                "Ticker":ticker,
                "Regime_Method":regime_method,
                "precision":precision,
                "max_dd":max_dd,
                "sharpe":sharpe

            }
        )
    def load_results(self,ticker,regime_method):
        output=self.results.find_one({
            "Ticker":ticker,
            "Regime_Method":regime_method
        })
        #find one returns a cursor (a string of documents)
        return output
    def save_regime(self,df,val):
        df.reset_index(inplace=True)
        regimes=df.to_dict(orient="records")
        #made it into a dictionary so it can be easily added.
        for regime in regimes:
            regime["Ticker"]=val
        self.regimes.insert_many(regimes)
    def load_regimes(self,val):
        regime=self.regimes.find({"Ticker":val})
        regime=pd.DataFrame(list(regime))
        if regime.empty:
            return regime
        regime.drop(["_id", "Ticker"], axis=1, inplace=True)
        regime.set_index("Date", inplace=True)
        return regime
    def save_predictions(self,df,val):
        df.reset_index(inplace=True)
        preds=df.to_dict(orient='records')
        for i in preds:
            i['Ticker']=val
        self.predictions.insert_many(preds)
    def load_predictions(self,val):
        pred=self.predictions.find({"Ticker":val})
        pred=pd.DataFrame(list(pred))
        if pred.empty:
            return pred
        pred.drop(["_id", "Ticker"], axis=1, inplace=True)
        pred.set_index("Date", inplace=True)

        return pred
    def load_forecast(self,ticker):
        out=self.forecast.find({"Ticker":ticker})
        out=pd.DataFrame(list(out))
        if out.empty:
            return out
        out.drop(['_id','Ticker'],axis=1,inplace=True)
        out.set_index("Date",inplace=True)
        return out
    def save_forecast(self,df,ticker):
        df.reset_index(inplace=True)
        input=df.to_dict(orient='records')
        for i in input:
            i['Ticker']=ticker
        self.forecast.insert_many(input)

    def save_shap(self, df, ticker, regime_method):
        df = df.reset_index()
        data = df.to_dict(orient="records")
        for rec in data:
            rec["Ticker"] = ticker
            rec["Regime_Method"] = regime_method
        self.SHAP.insert_many(data)

    def load_shap(self, ticker, regime_method):
        out = self.SHAP.find({"Ticker": ticker, "Regime_Method": regime_method})
        out = pd.DataFrame(list(out))
        if out.empty:
            return out
        out.drop(["_id", "Ticker", "Regime_Method"], axis=1, inplace=True)
        return out
