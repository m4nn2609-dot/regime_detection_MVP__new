from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", message="Invalid regime transition probabilities")
from features import reg_features
def fit_kmeans(df,n_regimes=2):
    #acts as a baseline(the most basic prediction technique)
    df[reg_features].dropna()
    kmeans=KMeans(n_clusters=n_regimes,random_state=42)
    kmeans.fit(df[reg_features])
    df['kmeans_labels']=kmeans.labels_
    return df
def fit_hmm(df,n_regimes=2):
    data=df[reg_features].dropna()
    hmm=MarkovRegression(data['Returns_5'],k_regimes=n_regimes,trend='c')
    pred=hmm.fit(disp=False, maxiter=1000, em_iter=50, search_reps=20)
    df['hmm_labels']=pred.smoothed_marginal_probabilities.idxmax(axis=1)
    return df
def fit_gmm(df,n_regimes=2):
    gmm=GaussianMixture(n_components=n_regimes)
    df['gmm_labels']=gmm.fit_predict(df[reg_features])
    return df


