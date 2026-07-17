from sklearn.metrics import precision_score

def evaluate(predictions, df):
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['returns'] = df['returns'] * predictions['Predictions'].shift(1)
    precision = precision_score(predictions['Target'], predictions['Predictions'])
    sharpe = df['returns'].mean() / df['returns'].std() * (252 ** 0.5)
    cumulative = (1 + df['returns']).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    return {"precision": precision, "sharpe": sharpe, "max_dd": max_dd}