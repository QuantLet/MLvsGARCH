import pandas as pd


def get_boxdata(cvtable, metric, classifiers, qw=(0.01, 24)):
    boxdata = []
    for m in classifiers:
        mdata = []
        for k in cvtable:
            mdata.append(cvtable[k][m][metric].loc[qw])
        boxdata.append(mdata)
    return boxdata


def get_daily_df(finalcomp):
    c = list(filter(lambda x: 'ret' in x, finalcomp.columns))
    daily_data = finalcomp[c]
    daily_data = daily_data.cumsum() + 1
    daily_data = daily_data.loc[pd.date_range(daily_data.index[0], daily_data.index[-1], freq='D')]
    daily_data = daily_data.pct_change().dropna()

    return daily_data


def get_col_from_wq(w, q):
    columns = {'drop': 'drop_%s_0.%s' % (w, q),
               'lower': 'lower_%s_0.%s' % (w, q),
               'proba_dl': 'proba_dl_%s_0.%s' % (w, q),
               'proba_lstm': 'proba_lstm_%s_0.%s' % (w, q),
               'proba_mlp': 'proba_mlp_%s_0.%s' % (w, q),
               'proba_norm': 'proba_norm_%s_0.%s' % (w, q),
               'proba_evt': 'proba_evt_%s_0.%s' % (w, q),
               'proba_carl': 'proba_carl_%s_0.%s' % (w, q),
               'proba_lpa': 'proba_lpa_norm_%s_0.%s' % (w, q),
               'proba_ensemble': 'proba_ensemble_%s_0.%s' % (w, q),
               'ret_dl': 'ret_dl_%s_0.%s' % (w, q),
               'ret_lstm': 'ret_lstm_%s_0.%s' % (w, q),
               'ret_mlp': 'ret_mlp_%s_0.%s' % (w, q),
               'ret_norm': 'ret_norm_%s_0.%s' % (w, q),
               'ret_evt': 'ret_evt_%s_0.%s' % (w, q),
               'ret_varspread': 'ret_varspread_%s_0.%s' % (w, q),
               'ret_carl': 'ret_carl_%s_0.%s' % (w, q),
               'ret_lpa': 'ret_lpa_norm_%s_0.%s' % (w, q),
               'ret_ensemble': 'ret_ensemble_%s_0.%s' % (w, q),
               'ret_var_norm': 'ret_var_norm_%s_0.%s' % (w, q),
               'ret_var_evt': 'ret_var_evt_%s_0.%s' % (w, q),
               'ret_switch': 'ret_switch_%s_0.%s' % (w, q),
               'pred_dl': 'pred_dl_%s_0.%s' % (w, q),
               'pred_lstm': 'pred_lstm_%s_0.%s' % (w, q),
               'pred_mlp': 'pred_mlp_%s_0.%s' % (w, q),
               'pred_norm': 'pred_norm_%s_0.%s' % (w, q),
               'pred_evt': 'pred_evt_%s_0.%s' % (w, q),
               'pred_varspread': 'label_varspread_%s_0.%s' % (w, q),
               'pred_carl': 'pred_carl_%s_0.%s' % (w, q),
               'pred_lpa': 'pred_lpa_norm_%s_0.%s' % (w, q),
               'pred_ensemble': 'pred_ensemble_%s_0.%s' % (w, q)
               }
    return columns


def qs_from_name(qs_name):
    qs = {}
    for k in qs_name:
        qs[k] = float('0.' + k)

    return qs
