import pandas as pd
import numpy as np
from sklearn import metrics
from result.utils import get_col_from_wq, get_daily_df

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc

WINDOWS = {'woneday': 24, 'wonemonth': 24 * 30, 'w4months': 24 * 30 * 4, 'w6months': 24 * 30 * 6, 'woneyear': 24 * 365}


def qs_from_name(qs_name):
    qs = {}
    for k in qs_name:
        qs[k] = float('0.' + k)

    return qs


def min_tpr_from_exceedance(exceedance, alpha):
    if exceedance == 0:
        return 0
    else:
        return (exceedance - alpha) / exceedance


def min_tpr_from_target(target, alpha):
    alpha_bar = sum(target == 1) / len(target)
    return min_tpr_from_exceedance(alpha_bar, alpha)


def return_fpr_tpr(target, pred, c01, c11):
    """

    :param target: target array
    :param pred: pred array
    :param c01: cost
    :param c11: cost
    :return:
    """
    cm = metrics.confusion_matrix(target, pred)
    class0 = sum(cm[0, :])
    class1 = sum(cm[1, :])

    p0 = class0 / (class0 + class1)
    p1 = 1 - p0
    FPR = cm[0, 1] / class0
    TPR = cm[1, 1] / class1

    return p0 * c01 * FPR, p1 * c11 * TPR


def get_risk_adjusted_roc(target, pred, returns):
    # cost FPR
    c01 = 1.0 * np.mean(returns.loc[target == 0])
    c11 = -1.0 * np.mean(returns.loc[target == 1])
    threshold = np.linspace(0, 1, 100)
    return_roc = np.array([[t, return_fpr_tpr(target, (pred >= t).astype(int), c01, c11)] for t in threshold])
    return_roc = np.array([[a[0], a[1][0], a[1][1]] for a in return_roc])

    return return_roc[:, 0], return_roc[:, 1], return_roc[:, 2]


def get_adjusted_auc(target, pred, returns):
    t, fpr, tpr = get_risk_adjusted_roc(target, pred, returns)
    scaled = np.zeros((fpr.shape[0], 2))
    scaled[:, 0] = fpr
    scaled[:, 1] = tpr
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(scaled)
    return auc(scaled[:, 0], scaled[:, 1])


def sharpe_ratio(returns, period=1, benchmark=0):
    return (returns - benchmark).mean() / (returns - benchmark).std() * np.sqrt(period)


def get_daily_finalcomp(finalcomp):
    c = list(filter(lambda x: 'ret' in x, finalcomp.columns))
    daily_data = finalcomp[c]
    daily_data = daily_data.cumsum() + 1
    daily_data = daily_data.loc[pd.date_range(daily_data.index[0], daily_data.index[-1], freq='D')]
    daily_data = daily_data.pct_change().dropna()

    return daily_data


def get_mdd(K):
    dd = K / K.cummax() - 1.0
    mdd = dd.cummin()
    mdd = abs(min(mdd))
    return mdd


def sortino_ratio(returns, period=1):
    # Create a downside return column with the negative returns only
    downside_returns = returns.loc[returns < 0]
    # Calculate expected return and std dev of downside
    expected_return = returns.mean()
    down_stdev = downside_returns.std()

    return expected_return / down_stdev * np.sqrt(period)


def get_table_report(finalcomp, qs_name, ws_name, mlp=False, varspread=False, carl=False, lpa=False, ensemble=False,
                     var_norm=False, var_evt=False, switch=False):
    qs = qs_from_name(qs_name)
    ws = WINDOWS
    # Get daily data
    daily_data = get_daily_df(finalcomp)

    # Build final performance table
    index = pd.MultiIndex.from_product([qs_name, ws_name], names=['alpha', 'window'])
    model_cols = ['btc', 'lstm', 'garch', 'evtgarch']

    if varspread:
        model_cols = model_cols + ['varspread']
    if carl:
        model_cols = model_cols + ['carl']
    if lpa:
        model_cols = model_cols + ['lpa']
    if ensemble:
        model_cols = model_cols + ['ensemble']
    if mlp:
        model_cols = model_cols + ['mlp']
    if var_norm:
        model_cols = model_cols + ['var_norm']
    if var_evt:
        model_cols = model_cols + ['var_evt']
    if switch:
        model_cols = model_cols + ['switch']

    columns = pd.MultiIndex.from_product([model_cols,
                                          ['ret', 'sr', 'exceedance', 'VaR', 'status', 'mdd', 'sortino']])
    table = pd.DataFrame(index=index,
                         columns=columns)

    ## total return
    print('total return')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'total_ret')] = np.cumsum(finalcomp['returns']).values[-1]
        table.loc[qw, ('lstm', 'total_ret')] = np.cumsum(finalcomp[c['ret_lstm']]).values[-1]
        table.loc[qw, ('garch', 'total_ret')] = np.cumsum(finalcomp[c['ret_norm']]).values[-1]
        table.loc[qw, ('evtgarch', 'total_ret')] = np.cumsum(finalcomp[c['ret_evt']]).values[-1]
        if varspread:
            table.loc[qw, ('varspread', 'total_ret')] = np.cumsum(finalcomp[c['ret_varspread']]).values[-1]
        if carl:
            table.loc[qw, ('carl', 'total_ret')] = np.cumsum(finalcomp[c['ret_carl']]).values[-1]
        if lpa:
            table.loc[qw, ('lpa', 'total_ret')] = np.cumsum(finalcomp[c['ret_lpa']]).values[-1]
        if ensemble:
            table.loc[qw, ('ensemble', 'total_ret')] = np.cumsum(finalcomp[c['ret_ensemble']]).values[-1]
        if mlp:
            table.loc[qw, ('mlp', 'total_ret')] = np.cumsum(finalcomp[c['ret_mlp']]).values[-1]
        if var_norm:
            table.loc[qw, ('var_norm', 'total_ret')] = np.cumsum(finalcomp[c['ret_var_norm']]).values[-1]
        if var_evt:
            table.loc[qw, ('var_evt', 'total_ret')] = np.cumsum(finalcomp[c['ret_var_evt']]).values[-1]
        if switch:
            table.loc[qw, ('switch', 'total_ret')] = np.cumsum(finalcomp[c['ret_switch']]).values[-1]

    print('Avg return')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'ret')] = np.mean(finalcomp['returns'])
        table.loc[qw, ('lstm', 'ret')] = np.mean(finalcomp[c['ret_lstm']])
        table.loc[qw, ('garch', 'ret')] = np.mean(finalcomp[c['ret_norm']])
        table.loc[qw, ('evtgarch', 'ret')] = np.mean(finalcomp[c['ret_evt']])
        if varspread:
            table.loc[qw, ('varspread', 'ret')] = np.mean(finalcomp[c['ret_varspread']])
        if carl:
            table.loc[qw, ('carl', 'ret')] = np.mean(finalcomp[c['ret_carl']])
        if lpa:
            table.loc[qw, ('lpa', 'ret')] = np.mean(finalcomp[c['ret_lpa']])
        if ensemble:
            table.loc[qw, ('ensemble', 'ret')] = np.mean(finalcomp[c['ret_ensemble']])
        if mlp:
            table.loc[qw, ('mlp', 'ret')] = np.mean(finalcomp[c['ret_mlp']])
        if var_norm:
            table.loc[qw, ('var_norm', 'ret')] = np.mean(finalcomp[c['ret_var_norm']])
        if var_evt:
            table.loc[qw, ('var_evt', 'ret')] = np.mean(finalcomp[c['ret_var_evt']])
        if switch:
            table.loc[qw, ('switch', 'ret')] = np.mean(finalcomp[c['ret_switch']])

    print('Avg return classes 0 and 1')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        for class_ in [0, 1]:
            table.loc[qw, ('btc', 'ret_%s' % class_)] = np.mean(
                finalcomp.loc[finalcomp[c['drop']] == class_, 'returns'])
            table.loc[qw, ('lstm', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                finalcomp.loc[finalcomp[c['pred_lstm']] == class_, 'returns'])
            table.loc[qw, ('garch', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                finalcomp.loc[finalcomp[c['pred_norm']] == class_, 'returns'])
            table.loc[qw, ('evtgarch', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                finalcomp.loc[finalcomp[c['pred_evt']] == class_, 'returns'])
            if carl:
                table.loc[qw, ('carl', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                    finalcomp.loc[finalcomp[c['pred_carl']] == class_, 'returns'])
            if lpa:
                table.loc[qw, ('lpa', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                    finalcomp.loc[finalcomp[c['pred_lpa']] == class_, 'returns'])
            if ensemble:
                table.loc[qw, ('ensemble', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                    finalcomp.loc[finalcomp[c['pred_ensemble']] == class_, 'returns'])
            if mlp:
                table.loc[qw, ('mlp', 'ret_%s' % class_)] = (-1.0) ** (class_ == 1) * np.mean(
                    finalcomp.loc[finalcomp[c['pred_mlp']] == class_, 'returns'])

    ## Volatility return
    print('Volatility')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'vol')] = np.std(finalcomp['returns'])
        table.loc[qw, ('lstm', 'vol')] = np.std(finalcomp[c['ret_lstm']])
        table.loc[qw, ('garch', 'vol')] = np.std(finalcomp[c['ret_norm']])
        table.loc[qw, ('evtgarch', 'vol')] = np.std(finalcomp[c['ret_evt']])
        if varspread:
            table.loc[qw, ('varspread', 'vol')] = np.std(finalcomp[c['ret_varspread']])
        if carl:
            table.loc[qw, ('carl', 'vol')] = np.std(finalcomp[c['ret_carl']])
        if lpa:
            table.loc[qw, ('lpa', 'vol')] = np.std(finalcomp[c['ret_lpa']])
        if ensemble:
            table.loc[qw, ('ensemble', 'vol')] = np.std(finalcomp[c['ret_ensemble']])
        if mlp:
            table.loc[qw, ('mlp', 'vol')] = np.std(finalcomp[c['ret_mlp']])
        if var_norm:
            table.loc[qw, ('var_norm', 'vol')] = np.std(finalcomp[c['ret_var_norm']])
        if var_evt:
            table.loc[qw, ('var_evt', 'vol')] = np.std(finalcomp[c['ret_var_evt']])
        if switch:
            table.loc[qw, ('switch', 'vol')] = np.std(finalcomp[c['ret_switch']])

    ## Sharpe ratio
    print('Excess sharpe ratio')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'excess_sr')] = sharpe_ratio(finalcomp['returns'],
                                                           period=365 * 24)

        table.loc[qw, ('lstm', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_lstm']],
                                                            period=365 * 24,
                                                            benchmark=finalcomp['returns'])

        table.loc[qw, ('garch', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_norm']],
                                                             period=365 * 24,
                                                             benchmark=finalcomp['returns'])

        table.loc[qw, ('evtgarch', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_evt']],
                                                                period=365 * 24,
                                                                benchmark=finalcomp['returns'])
        if varspread:
            table.loc[qw, ('varspread', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_varspread']],
                                                                     period=365 * 24,
                                                                     benchmark=finalcomp['returns'])
        if carl:
            table.loc[qw, ('carl', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_carl']],
                                                                period=365 * 24,
                                                                benchmark=finalcomp['returns'])
        if lpa:
            table.loc[qw, ('lpa', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_lpa']],
                                                               period=365 * 24,
                                                               benchmark=finalcomp['returns'])
        if ensemble:
            table.loc[qw, ('ensemble', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_ensemble']],
                                                                    period=365 * 24,
                                                                    benchmark=finalcomp['returns'])
        if mlp:
            table.loc[qw, ('mlp', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_mlp']],
                                                               period=365 * 24,
                                                               benchmark=finalcomp['returns'])
        if var_norm:
            table.loc[qw, ('var_norm', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_var_norm']],
                                                                    period=365 * 24,
                                                                    benchmark=finalcomp['returns'])
        if var_evt:
            table.loc[qw, ('var_evt', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_var_evt']],
                                                                   period=365 * 24,
                                                                   benchmark=finalcomp['returns'])
        if switch:
            table.loc[qw, ('switch', 'excess_sr')] = sharpe_ratio(finalcomp[c['ret_switch']],
                                                                  period=365 * 24,
                                                                  benchmark=finalcomp['returns'])
    print('Sharpe ratio')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'sr')] = sharpe_ratio(finalcomp['returns'],
                                                    period=365 * 24)
        table.loc[qw, ('lstm', 'sr')] = sharpe_ratio(finalcomp[c['ret_lstm']],
                                                     period=365 * 24)
        table.loc[qw, ('garch', 'sr')] = sharpe_ratio(finalcomp[c['ret_norm']],
                                                      period=365 * 24)
        table.loc[qw, ('evtgarch', 'sr')] = sharpe_ratio(finalcomp[c['ret_evt']],
                                                         period=365 * 24)
        if varspread:
            table.loc[qw, ('varspread', 'sr')] = sharpe_ratio(finalcomp[c['ret_varspread']],
                                                              period=365 * 24)
        if carl:
            table.loc[qw, ('carl', 'sr')] = sharpe_ratio(finalcomp[c['ret_carl']],
                                                         period=365 * 24)
        if lpa:
            table.loc[qw, ('lpa', 'sr')] = sharpe_ratio(finalcomp[c['ret_lpa']],
                                                        period=365 * 24)
        if ensemble:
            table.loc[qw, ('ensemble', 'sr')] = sharpe_ratio(finalcomp[c['ret_ensemble']],
                                                             period=365 * 24)
        if mlp:
            table.loc[qw, ('mlp', 'sr')] = sharpe_ratio(finalcomp[c['ret_mlp']],
                                                        period=365 * 24)
        if var_norm:
            table.loc[qw, ('var_norm', 'sr')] = sharpe_ratio(finalcomp[c['ret_var_norm']],
                                                             period=365 * 24)
        if var_evt:
            table.loc[qw, ('var_evt', 'sr')] = sharpe_ratio(finalcomp[c['ret_var_evt']],
                                                            period=365 * 24)
        if switch:
            table.loc[qw, ('switch', 'sr')] = sharpe_ratio(finalcomp[c['ret_switch']],
                                                           period=365 * 24)

    ## Exceedance
    print('Exceedance')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'exceedance')] = finalcomp[c['drop']].sum() / len(finalcomp)
        cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_lstm']])
        table.loc[qw, ('lstm', 'exceedance')] = cm[1, 0] / np.sum(cm)
        cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_norm']])
        table.loc[qw, ('garch', 'exceedance')] = cm[1, 0] / np.sum(cm)
        cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_evt']])
        table.loc[qw, ('evtgarch', 'exceedance')] = cm[1, 0] / np.sum(cm)
        if varspread:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_varspread']])
            table.loc[qw, ('varspread', 'exceedance')] = cm[1, 0] / np.sum(cm)
        if carl:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_carl']])
            table.loc[qw, ('carl', 'exceedance')] = cm[1, 0] / np.sum(cm)
        if lpa:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_lpa']])
            table.loc[qw, ('lpa', 'exceedance')] = cm[1, 0] / np.sum(cm)
        if ensemble:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_ensemble']])
            table.loc[qw, ('ensemble', 'exceedance')] = cm[1, 0] / np.sum(cm)
        if mlp:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_mlp']])
            table.loc[qw, ('mlp', 'exceedance')] = cm[1, 0] / np.sum(cm)
        if var_norm:
            table.loc[qw, ('var_norm', 'exceedance')] = np.sum(
                finalcomp[c['ret_var_norm']] < finalcomp[c['lower']]) / len(finalcomp)
        if var_evt:
            table.loc[qw, ('var_evt', 'exceedance')] = np.sum(
                finalcomp[c['ret_var_evt']] < finalcomp[c['lower']]) / len(finalcomp)
        if switch:
            table.loc[qw, ('switch', 'exceedance')] = np.sum(
                finalcomp[c['ret_switch']] < finalcomp[c['lower']]) / len(finalcomp)

    # Value-At-Risk and status
    print('Value-At-Risk and status')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'VaR')] = - np.quantile(finalcomp['returns'], qs[qw[0]])
        table.loc[qw, ('btc', 'status')] = np.quantile(finalcomp['returns'], qs[qw[0]]) > np.quantile(
            finalcomp['returns'], qs[qw[0]])

        table.loc[qw, ('lstm', 'VaR')] = - np.quantile(finalcomp[c['ret_lstm']], qs[qw[0]])
        table.loc[qw, ('lstm', 'status')] = np.quantile(finalcomp[c['ret_lstm']], qs[qw[0]]) > np.quantile(
            finalcomp['returns'], qs[qw[0]])

        table.loc[qw, ('garch', 'VaR')] = - np.quantile(finalcomp[c['ret_norm']], qs[qw[0]])
        table.loc[qw, ('garch', 'status')] = np.quantile(finalcomp[c['ret_norm']], qs[qw[0]]) > np.quantile(
            finalcomp['returns'], qs[qw[0]])

        table.loc[qw, ('evtgarch', 'VaR')] = - np.quantile(finalcomp[c['ret_evt']], qs[qw[0]])
        table.loc[qw, ('evtgarch', 'status')] = np.quantile(finalcomp[c['ret_evt']], qs[qw[0]]) > np.quantile(
            finalcomp['returns'], qs[qw[0]])

        if varspread:
            table.loc[qw, ('varspread', 'VaR')] = - np.quantile(finalcomp[c['ret_varspread']], qs[qw[0]])
            table.loc[qw, ('varspread', 'status')] = np.quantile(finalcomp[c['ret_varspread']],
                                                                 qs[qw[0]]) > np.quantile(finalcomp['returns'],
                                                                                          qs[qw[0]])
        if carl:
            table.loc[qw, ('carl', 'VaR')] = - np.quantile(finalcomp[c['ret_carl']], qs[qw[0]])
            table.loc[qw, ('carl', 'status')] = np.quantile(finalcomp[c['ret_carl']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])
        if lpa:
            table.loc[qw, ('lpa', 'VaR')] = - np.quantile(finalcomp[c['ret_lpa']], qs[qw[0]])
            table.loc[qw, ('lpa', 'status')] = np.quantile(finalcomp[c['ret_lpa']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])
        if ensemble:
            table.loc[qw, ('ensemble', 'VaR')] = - np.quantile(finalcomp[c['ret_ensemble']], qs[qw[0]])
            table.loc[qw, ('ensemble', 'status')] = np.quantile(finalcomp[c['ret_ensemble']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])
        if mlp:
            table.loc[qw, ('mlp', 'VaR')] = - np.quantile(finalcomp[c['ret_mlp']], qs[qw[0]])
            table.loc[qw, ('mlp', 'status')] = np.quantile(finalcomp[c['ret_mlp']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])
        if var_norm:
            table.loc[qw, ('var_norm', 'VaR')] = - np.quantile(finalcomp[c['ret_var_norm']], qs[qw[0]])
            table.loc[qw, ('var_norm', 'status')] = np.quantile(finalcomp[c['ret_var_norm']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])
        if var_evt:
            table.loc[qw, ('var_evt', 'VaR')] = - np.quantile(finalcomp[c['ret_var_evt']], qs[qw[0]])
            table.loc[qw, ('var_evt', 'status')] = np.quantile(finalcomp[c['ret_var_evt']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])
        if switch:
            table.loc[qw, ('switch', 'VaR')] = - np.quantile(finalcomp[c['ret_switch']], qs[qw[0]])
            table.loc[qw, ('switch', 'status')] = np.quantile(finalcomp[c['ret_switch']], qs[qw[0]]) > np.quantile(
                finalcomp['returns'], qs[qw[0]])

    # MDD
    print('Max DrawDown')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'mdd')] = get_mdd((finalcomp['returns'] + 1).cumprod())
        table.loc[qw, ('lstm', 'mdd')] = get_mdd((finalcomp[c['ret_lstm']] + 1).cumprod())
        table.loc[qw, ('garch', 'mdd')] = get_mdd((finalcomp[c['ret_norm']] + 1).cumprod())
        table.loc[qw, ('evtgarch', 'mdd')] = get_mdd((finalcomp[c['ret_evt']] + 1).cumprod())
        if varspread:
            table.loc[qw, ('varspread', 'mdd')] = get_mdd((finalcomp[c['ret_varspread']] + 1).cumprod())
        if carl:
            table.loc[qw, ('carl', 'mdd')] = get_mdd((finalcomp[c['ret_carl']] + 1).cumprod())
        if lpa:
            table.loc[qw, ('lpa', 'mdd')] = get_mdd((finalcomp[c['ret_lpa']] + 1).cumprod())
        if ensemble:
            table.loc[qw, ('ensemble', 'mdd')] = get_mdd((finalcomp[c['ret_ensemble']] + 1).cumprod())
        if mlp:
            table.loc[qw, ('mlp', 'mdd')] = get_mdd((finalcomp[c['ret_mlp']] + 1).cumprod())
        if var_norm:
            table.loc[qw, ('var_norm', 'mdd')] = get_mdd((finalcomp[c['ret_var_norm']] + 1).cumprod())
        if var_evt:
            table.loc[qw, ('var_evt', 'mdd')] = get_mdd((finalcomp[c['ret_var_evt']] + 1).cumprod())
        if switch:
            table.loc[qw, ('switch', 'mdd')] = get_mdd((finalcomp[c['ret_switch']] + 1).cumprod())

    # Sortino ratio
    print('Sortino ratio')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'sortino')] = sortino_ratio(daily_data['returns'], period=365)
        table.loc[qw, ('lstm', 'sortino')] = sortino_ratio(daily_data[c['ret_lstm']], period=365)
        table.loc[qw, ('garch', 'sortino')] = sortino_ratio(daily_data[c['ret_norm']], period=365)
        table.loc[qw, ('evtgarch', 'sortino')] = sortino_ratio(daily_data[c['ret_evt']], period=365)
        if varspread:
            table.loc[qw, ('varspread', 'sortino')] = sortino_ratio(daily_data[c['ret_varspread']], period=365)
        if carl:
            table.loc[qw, ('carl', 'sortino')] = sortino_ratio(daily_data[c['ret_carl']], period=365)
        if lpa:
            table.loc[qw, ('lpa', 'sortino')] = sortino_ratio(daily_data[c['ret_lpa']], period=365)
        if ensemble:
            table.loc[qw, ('ensemble', 'sortino')] = sortino_ratio(daily_data[c['ret_ensemble']], period=365)
        if mlp:
            table.loc[qw, ('mlp', 'sortino')] = sortino_ratio(daily_data[c['ret_mlp']], period=365)
        if var_norm:
            table.loc[qw, ('var_norm', 'sortino')] = sortino_ratio(daily_data[c['ret_var_norm']], period=365)
        if var_evt:
            table.loc[qw, ('var_evt', 'sortino')] = sortino_ratio(daily_data[c['ret_var_evt']], period=365)
        if switch:
            table.loc[qw, ('switch', 'sortino')] = sortino_ratio(daily_data[c['ret_switch']], period=365)

    # Classification
    # Brier score
    print('Brier score')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('lstm', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']], finalcomp[c['proba_lstm']])
        table.loc[qw, ('garch', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']], finalcomp[c['proba_norm']])
        table.loc[qw, ('evtgarch', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']], finalcomp[c['proba_evt']])
        if carl:
            table.loc[qw, ('carl', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']],
                                                                        finalcomp[c['proba_carl']])
        if lpa:
            table.loc[qw, ('lpa', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']], finalcomp[c['proba_lpa']])
        if ensemble:
            table.loc[qw, ('ensemble', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']],
                                                                            finalcomp[c['proba_ensemble']])
        if mlp:
            table.loc[qw, ('mlp', 'brier')] = metrics.brier_score_loss(finalcomp[c['drop']], finalcomp[c['proba_mlp']])

    # Cross entropy
    print('Cross entropy')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('lstm', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']], finalcomp[c['proba_lstm']])
        table.loc[qw, ('garch', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']], finalcomp[c['proba_norm']])
        table.loc[qw, ('evtgarch', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']], finalcomp[c['proba_evt']])
        if carl:
            table.loc[qw, ('carl', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']], finalcomp[c['proba_carl']])
        if lpa:
            table.loc[qw, ('lpa', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']], finalcomp[c['proba_lpa']])
        if ensemble:
            table.loc[qw, ('ensemble', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']],
                                                                       finalcomp[c['proba_ensemble']])
        if mlp:
            table.loc[qw, ('mlp', 'log_loss')] = metrics.log_loss(finalcomp[c['drop']], finalcomp[c['proba_mlp']])

    # TPR, TNR
    print('TPR and TNR')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('btc', 'tpr')] = min_tpr_from_target(finalcomp[c['drop']], float('0.%s' % qw[0]))
        cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_lstm']])
        table.loc[qw, ('lstm', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
        table.loc[qw, ('lstm', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_norm']])
        table.loc[qw, ('garch', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
        table.loc[qw, ('garch', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_evt']])
        table.loc[qw, ('evtgarch', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
        table.loc[qw, ('evtgarch', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        if varspread:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_varspread']])
            table.loc[qw, ('varspread', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
            table.loc[qw, ('varspread', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        if carl:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_carl']])
            table.loc[qw, ('carl', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
            table.loc[qw, ('carl', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        if lpa:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_lpa']])
            table.loc[qw, ('lpa', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
            table.loc[qw, ('lpa', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        if ensemble:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_ensemble']])
            table.loc[qw, ('ensemble', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
            table.loc[qw, ('ensemble', 'tnr')] = cm[0, 0] / cm.sum(1)[0]
        if mlp:
            cm = metrics.confusion_matrix(finalcomp[c['drop']], finalcomp[c['pred_ensemble']])
            table.loc[qw, ('mlp', 'tpr')] = cm[1, 1] / cm.sum(1)[-1]
            table.loc[qw, ('mlp', 'tnr')] = cm[0, 0] / cm.sum(1)[0]

    # F score
    print('F score')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('lstm', 'fscore')] = metrics.f1_score(finalcomp[c['drop']], finalcomp[c['pred_lstm']])
        table.loc[qw, ('garch', 'fscore')] = metrics.f1_score(finalcomp[c['drop']], finalcomp[c['pred_norm']])
        table.loc[qw, ('evtgarch', 'fscore')] = metrics.f1_score(finalcomp[c['drop']], finalcomp[c['pred_evt']])
        if varspread:
            table.loc[qw, ('varspread', 'fscore')] = metrics.f1_score(finalcomp[c['drop']],
                                                                      finalcomp[c['pred_varspread']])
        if carl:
            table.loc[qw, ('carl', 'fscore')] = metrics.f1_score(finalcomp[c['drop']], finalcomp[c['pred_carl']])
        if lpa:
            table.loc[qw, ('lpa', 'fscore')] = metrics.f1_score(finalcomp[c['drop']], finalcomp[c['pred_lpa']])
        if ensemble:
            table.loc[qw, ('ensemble', 'fscore')] = metrics.f1_score(finalcomp[c['drop']],
                                                                     finalcomp[c['pred_ensemble']])
        if mlp:
            table.loc[qw, ('mlp', 'fscore')] = metrics.f1_score(finalcomp[c['drop']], finalcomp[c['pred_mlp']])

    # AUC
    print('AUC score')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])
        table.loc[qw, ('lstm', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']], finalcomp[c['proba_lstm']])
        table.loc[qw, ('garch', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']], finalcomp[c['proba_norm']])
        table.loc[qw, ('evtgarch', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']], finalcomp[c['proba_evt']])
        if carl:
            table.loc[qw, ('carl', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']], finalcomp[c['proba_carl']])
        if lpa:
            table.loc[qw, ('lpa', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']], finalcomp[c['proba_lpa']])
        if ensemble:
            table.loc[qw, ('ensemble', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']],
                                                                       finalcomp[c['proba_ensemble']])
        if mlp:
            table.loc[qw, ('mlp', 'auc')] = metrics.roc_auc_score(finalcomp[c['drop']], finalcomp[c['proba_mlp']])

    # Risk-adjusted AUC
    print('Risk-adjusted AUC')
    for qw in table.index:
        c = get_col_from_wq(qw[1], qw[0])

        table.loc[qw, ('lstm', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']], finalcomp[c['proba_lstm']],
                                                               finalcomp['returns'])
        table.loc[qw, ('garch', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']], finalcomp[c['proba_norm']],
                                                                finalcomp['returns'])
        table.loc[qw, ('evtgarch', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']], finalcomp[c['proba_evt']],
                                                                   finalcomp['returns'])
        if carl:
            table.loc[qw, ('carl', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']],
                                                                   finalcomp[c['proba_carl']], finalcomp['returns'])
        if lpa:
            table.loc[qw, ('lpa', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']], finalcomp[c['proba_lpa']],
                                                                  finalcomp['returns'])
        if ensemble:
            table.loc[qw, ('ensemble', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']],
                                                                       finalcomp[c['proba_ensemble']],
                                                                       finalcomp['returns'])
        if mlp:
            table.loc[qw, ('mlp', 'risk_auc')] = get_adjusted_auc(finalcomp[c['drop']], finalcomp[c['proba_mlp']],
                                                                  finalcomp['returns'])

    # Respected constraint
    print('Respected constraint')
    lc = ['lstm', 'garch', 'evtgarch']
    if varspread:
        lc = lc + ['varspread']
    if carl:
        lc = lc + ['carl']
    if lpa:
        lc = lc + ['lpa']
    if ensemble:
        lc = lc + ['ensemble']
    if mlp:
        lc = lc + ['mlp']
    for c in lc:
        table[(c, 'respected_cstr')] = table.loc[:, 'btc']['tpr'] <= table.loc[:, c]['tpr']

    # Rearange columns order
    table.sort_index(axis=1, level=0, inplace=True)

    new_ind = pd.MultiIndex.from_product([[0.01, 0.025, 0.05, 0.1], [24, 2880, 4320]])
    table.index = new_ind

    return table
