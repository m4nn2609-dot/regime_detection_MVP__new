import pandas as pd
import prophet as prp
def forecast_price(df,days=30):
    ndf=pd.DataFrame({"ds":df.index,"y":df["Close"]})
    #prophet requires you to have column names as y and df only.
    model=prp.Prophet()
    model.fit(ndf)
    future=model.make_future_dataframe(periods=days)
    forecast=model.predict(future)
    return forecast



