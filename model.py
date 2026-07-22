import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def train_regime_model(df, predictors, regime_col="kmeans_labels"):
    output = {}
    for regime_id, group in df.groupby(regime_col):
        if len(group)<30:
            print(f"Regime {regime_id}: too little data, skipping.")
            continue
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(group[predictors], group["Target"])
        output[regime_id]=model
    return output
def predict_with_regime(test_df, regime_models, predictors, regime_col="kmeans_labels"):
    probs=[]
    indices=[]
    for index, row in test_df.iterrows():
        regime=row[regime_col]
        if regime not in regime_models:
            continue
        model=regime_models[regime]
        prob = model.predict_proba(row[predictors].to_frame().T)[:, 1]
        probs.append(prob[0])
        indices.append(index)
    predictions=(np.array(probs)>=0.5).astype(int)
    result = pd.Series(predictions,index=indices,name="Predictions")
    result.index.name = test_df.index.name  # preserve "Date"
    return result
def backtest_regime(df,predictors,regime_col="kmeans_labels",n_splits=5,gap=60):
    tscv=TimeSeriesSplit(n_splits=n_splits,gap=gap)
    results=[]
    for train_idx,test_idx in tscv.split(df):
        train=df.iloc[train_idx]
        test=df.iloc[test_idx]
        res=train_regime_model(train,predictors,regime_col=regime_col)
        fin_preds=predict_with_regime(test,res,predictors,regime_col)
        combined=pd.concat([test['Target'],fin_preds],axis=1).dropna()
        results.append(combined)
    return pd.concat(results)
def label_regimes(df, label_col):
    stats = df.groupby(label_col)['Returns_5'].agg(['mean', 'std']).sort_values('mean')
    stats['regime_name'] = "Sideways"
    if len(stats) >= 2:
        stats.iloc[0, stats.columns.get_loc('regime_name')] = "Bear"
        stats.iloc[-1, stats.columns.get_loc('regime_name')] = "Bull"
    most_volatile = stats['std'].idxmax()
    stats.loc[most_volatile, 'regime_name'] = "High Volatility"
    mapping = stats['regime_name'].to_dict()
    df[f"{label_col}_name"] = df[label_col].map(mapping)
    return df