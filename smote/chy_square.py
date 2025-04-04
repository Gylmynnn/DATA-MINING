from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd


def chi_square(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series ) -> SelectKBest:
    chi_selector = SelectKBest(score_func=chi2, k=8)
    return chi_selector.fit(X, y)
