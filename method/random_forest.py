from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from typing import Tuple, List
import pandas as pd


def random_forest (X : pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> Tuple[float, List]:
    RF = RandomForestClassifier(n_estimators=100, random_state= 0)
    predict = cross_val_predict(RF, X, y, cv=10)
    accuracy =  metrics.accuracy_score(y, predict)
    return accuracy, predict



