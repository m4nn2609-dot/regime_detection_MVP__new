import pandas as pd
from sklearn.metrics import precision_score

def evaluate(predictions, df, benchmark_days=504):  # ~2 years of trading days
    if predictions is None or predictions.empty or predictions['Target'].nunique() < 2:
        return {"precision": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "strategy_return": 0.0, "buy_hold_return": 0.0, "excess_return": 0.0}

    df = df.copy()
    df['returns'] = df['Close'].pct_change().clip(-0.5, 0.5)
    df['strategy_returns'] = df['returns'] * predictions['Predictions'].shift(1)

    precision = precision_score(predictions['Target'], predictions['Predictions'], zero_division=0)
    std = df['strategy_returns'].std()
    sharpe = (df['strategy_returns'].mean() / std * (252 ** 0.5)) if std and not pd.isna(std) else 0.0

    strategy_cum_full = (1 + df['strategy_returns'].fillna(0)).cumprod()
    max_dd = (strategy_cum_full / strategy_cum_full.cummax() - 1).min()

    # restrict return comparison to the most recent window only
    recent_idx = predictions.index[-benchmark_days:] if len(predictions.index) > benchmark_days else predictions.index
    recent_strategy_returns = df.loc[recent_idx, 'strategy_returns']
    recent_actual_returns = df.loc[recent_idx, 'returns']

    strategy_cum = (1 + recent_strategy_returns.fillna(0)).cumprod()
    buy_hold_cum = (1 + recent_actual_returns.fillna(0)).cumprod()

    strategy_return = strategy_cum.iloc[-1] - 1 if len(strategy_cum) else 0.0
    buy_hold_return = buy_hold_cum.iloc[-1] - 1 if len(buy_hold_cum) else 0.0
    excess_return = strategy_return - buy_hold_return

    return {"precision": precision, "sharpe": sharpe, "max_dd": max_dd,
            "strategy_return": strategy_return, "buy_hold_return": buy_hold_return,
            "excess_return": excess_return}