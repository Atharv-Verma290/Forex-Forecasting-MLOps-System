import pandas as pd


def get_predictors(df: pd.DataFrame):
    vol_cols = [c for c in df.columns if 'vol_absmean_' in c]
    hl_mean_cols = [c for c in df.columns if 'hl_range_mean_' in c]
    cols_to_drop = [
        "open", "high", "low", "close", 
        "tomorrow", "target", 
        "log_return", "abs_log_return", 
        "hl_range", "large_move"
    ] + vol_cols + hl_mean_cols
    predictors = df.columns.drop(cols_to_drop, errors='ignore')
    return list(predictors)