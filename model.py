import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def train_regime_model(df, predictors, regime_col="kmeans_labels"):
    output = {}
    for regime_id, group in df.groupby(regime_col):
        if len(group)<100:
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
        prob=model.predict_proba(row[predictors].values.reshape(1, -1))[:, 1]
        probs.append(prob[0])
        indices.append(index)
    predictions=(np.array(probs)>=0.5).astype(int)
    return pd.Series(predictions,index=indices,name="Predictions")
def backtest_regime(df,predictors,regime_col="kmeans_labels",n_splits=5,gap=60):
    tscv=TimeSeriesSplit(n_splits=n_splits,gap=gap)
    results=[]
    for train_idx,test_idx in tscv.split(df):
        train=df.iloc[train_idx]
        test=df.iloc[test_idx]
        res=train_regime_model(train,predictors,regime_col=regime_col)
        fin_preds=predict_with_regime(test,res,predictors,regime_col)
        combined=pd.concat([test['Target'],fin_preds],axis=1)
        results.append(combined)
    return pd.concat(results)
