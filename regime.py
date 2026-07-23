from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", message="Invalid regime transition probabilities")
from features import reg_features
def fit_kmeans(df,n_regimes=4):
    #acts as a baseline(the most basic prediction technique)
    df[reg_features].dropna(inplace=True)
    kmeans=KMeans(n_clusters=n_regimes,random_state=42)
    kmeans.fit(df[reg_features])
    df['kmeans_labels']=kmeans.labels_
    return df
def fit_hmm(df,n_regimes=4):
    data=df[reg_features].dropna()
    hmm=MarkovRegression(data['Returns_5'],k_regimes=n_regimes,trend='c')
    pred=hmm.fit(disp=False, maxiter=1000, em_iter=50, search_reps=20)
    df['hmm_labels']=pred.smoothed_marginal_probabilities.idxmax(axis=1)
    return df
def fit_gmm(df,n_regimes=4):
    gmm=GaussianMixture(n_components=n_regimes)
    df['gmm_labels']=gmm.fit_predict(df[reg_features])
    return df
def forecast_regime(df, forecast_df, label_col):
    stats = df.groupby(label_col)['Returns_5'].agg(['mean'])
    name_col = f"{label_col}_name"
    name_map = df.drop_duplicates(label_col).set_index(label_col)[name_col].to_dict() if name_col in df.columns else {}

    cutoff = df.index.max()
    if getattr(cutoff, "tzinfo", None) is not None:
        cutoff = cutoff.tz_localize(None)

    future_only = forecast_df[forecast_df.index > cutoff].copy()
    future_only['Forecast_Returns_5'] = future_only['yhat'].pct_change(5)

    def closest_regime(x):
        if pd.isna(x):
            return None
        return (stats['mean'] - x).abs().idxmin()

    future_only['regime_id'] = future_only['Forecast_Returns_5'].apply(closest_regime)
    future_only[f"{label_col}_forecast_name"] = future_only['regime_id'].map(name_map)
    return future_only[[f"{label_col}_forecast_name"]]