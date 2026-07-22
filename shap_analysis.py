import shap
import pandas as pd

def compute_shap_for_regime_models(regime_models, df, predictors, regime_col):
    all_shap = []
    for regime_id, model in regime_models.items():
        subset = df[df[regime_col] == regime_id]
        if subset.empty:
            continue
        X = subset[predictors]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        shap_df = pd.DataFrame(shap_values, columns=predictors, index=X.index)
        shap_df["Regime"] = regime_id
        all_shap.append(shap_df)

    if not all_shap:
        return pd.DataFrame()
    return pd.concat(all_shap)