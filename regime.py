from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

from features import reg_features
def fit_kmeans(df,n_regimes=4):
    #acts as a baseline(the most basic prediction technique)
    df[reg_features].dropna()
    kmeans=KMeans(n_clusters=n_regimes,random_state=42)
    kmeans.fit(df[reg_features])
    df['KMeans_labels']=kmeans.labels_
    return df
def fit_hmm(df,n_regimes=4):
    data=df[reg_features].dropna()
    hmm=MarkovSwitching(data['Returns'],k_regimes=n_regimes,trend='c')
    pred=hmm.fit(disp=False)
    df['HMM_labels']=pred.smoothed_marginal_probabilities.idxmax(axis=1)
    return df
def fit_gmm(df,n_regimes=4):
    gmm=GaussianMixture(n_components=n_regimes)
    df['GMM_Labels']=gmm.fit(df[regime_features]).dropna()
    return df


