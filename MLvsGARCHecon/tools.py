import pandas as pd
import datetime as dt


def load_econ_pred(path='./saved_models/20190922155918or_prediction_10per_proba.csv', qs = [0.9]):
    df = pd.read_csv(path, index_col = 0)
    df.index = pd.to_datetime(df.index)
    # reindex by close time
    df.index = df.index + dt.timedelta(hours = 1)

    # Compute var - u and var breaks
    q_columns = ['threshold', 'evt_var', 'evt_es', 'var', 'es', 'mean', 'zq']
    df_columns = []
    for q in qs:
        df_columns.append([c + '_' + str(q) for c in q_columns])
    df_columns = [c for cl in df_columns for c in cl]

    for c in df_columns + ['std_losses']:
        df.loc[:, c] = - df.loc[:, c] * df['norm_sd']

    for c in ['sd_' + str(q) for q in qs]:
        df.loc[:, c] = df.loc[:, c] * df['norm_sd']

    df.columns = ['returns'] + list(df.columns)[1:]
    
    return df

