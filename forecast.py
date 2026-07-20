import pandas as pd
import prophet as prp

def forecast_price(df, days=30):
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    ndf = pd.DataFrame({"ds": idx, "y": df["Close"].values})
    model = prp.Prophet()
    model.fit(ndf)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    forecast = forecast.rename(columns={"ds": "Date"}).set_index("Date")
    return forecast



