# RMSEを計算する関数

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def check_rmse(y_pred):
    """
    RMSEを計算する関数
    Args:
        y_pred: 予測値
    Returns:
        RMSE
    """
    true_df = pd.read_csv('../data/treated/use/test_y.csv')
    rmse = np.sqrt(mean_squared_error(true_df, y_pred))
    print(f"RMSE: {rmse}")
    return rmse

