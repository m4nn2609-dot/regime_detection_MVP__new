import ta.volume
from ta import volatility, trend, momentum


def create_forward(df):
    df["Forward"] = df['Close'].shift(-5)
    return df
reg_features=['Returns_5', 'High_Low_Range', 'Gap']

def create_target(df):
    df['Target'] = (df['Forward'] > df['Close']).astype(int)
    return df


def add_features(df):
    rsi = momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["RSI"] = rsi
    s_rsi = momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
    df["StochR"] = s_rsi
    mac_d = trend.MACD(close=df['Close']).macd()
    df["MACD"] = mac_d
    b_b = volatility.BollingerBands(close=df['Close'], window=20)
    df["Bollinger_high"] = b_b.bollinger_hband()
    df['Bollinger_width'] = b_b.bollinger_wband()
    df['Bollinger_low'] = b_b.bollinger_lband()
    atr = volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['AverageTrueRange'] = atr
    lagged_features = ["AverageTrueRange", "RSI", "MACD"]
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df['Volume']).on_balance_volume()
    for feature in lagged_features:
        for lag in [1, 2, 3]:
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

    df["Williams %R"] = momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"]).williams_r()
    df["CCI"] = trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"]).cci()
    df["Vol_MA20"] = df['Volume'].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]
    df["Vol_Change"] = df["Volume"].pct_change().astype(float)

    up_day = (df['Close'] > df['Close'].shift(1)).astype(int)

    horizons = [2, 5, 60, 250]
    for horizon in horizons:
        rolling_close = df['Close'].rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df['Close'] / rolling_close
        df[f"Trend_Ratio_{horizon}"] = up_day.rolling(horizon).mean()

    df["Returns"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Returns"].rolling(20).std()
    df["Volatility_5"] = df["Returns"].rolling(5).std()
    return df
def regime_features(df):
    df['Returns_5']=df['Returns'].rolling(5).mean()
    df['High_Low_Range']=(df['High']-df['Low'])/df['Close']
    df['Gap']=(df['Open']-df['Close'].shift(1))/df['Close'].shift(1)
    return df

