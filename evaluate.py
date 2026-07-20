import pandas as pd
from sklearn.metrics import precision_score

def evaluate(predictions, df):
    if predictions is None or predictions.empty or predictions['Target'].nunique() < 2:
        return {"precision": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['returns'] = df['returns'] * predictions['Predictions'].shift(1)
    precision = precision_score(predictions['Target'], predictions['Predictions'], zero_division=0)
    std = df['returns'].std()
    sharpe = (df['returns'].mean() / std * (252 ** 0.5)) if std and not pd.isna(std) else 0.0
    cumulative = (1 + df['returns'].fillna(0)).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    return {"precision": precision, "sharpe": sharpe, "max_dd": max_dd}