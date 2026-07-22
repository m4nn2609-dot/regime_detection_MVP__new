import pandas as pd
import prophet as prp

def forecast_price(df, days=30, lookback_days=500):
    recent = df.tail(lookback_days)
    idx = recent.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    ndf = pd.DataFrame({"ds": idx, "y": recent["Close"].values})
    model = prp.Prophet(daily_seasonality=False)
    model.fit(ndf)
    future = model.make_future_dataframe(periods=days,freq="B")
    forecast = model.predict(future)
    forecast = forecast.rename(columns={"ds": "Date"}).set_index("Date")
    return forecast


