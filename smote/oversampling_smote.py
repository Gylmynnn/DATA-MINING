from imblearn.over_sampling import SMOTE
import pandas as pd
from typing import Tuple


def over_sampling_smote(X: pd.DataFrame ,y: pd.DataFrame | pd.Series) -> Tuple[
    pd.DataFrame | pd.Series, pd.DataFrame | pd.Series
]:
    smote = SMOTE()
    result = smote.fit_resample(X, y)
    x_smote, y_smote = result[0], result[1]
    return x_smote, y_smote




