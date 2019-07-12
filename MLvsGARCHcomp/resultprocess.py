import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import metrics


def get_pred(returns, dl_model, econ_model, threshold, cv_split_i, model_epoch, cv_split, n_classes):
    lstm_path = '%s/%s/e=%s-%s-prediction.pkl' % (dl_model, cv_split_i, model_epoch, cv_split)
    garch_path = "%s/cv%s.csv" % (econ_model, cv_split_i)
    prediction = pickle.load(open(lstm_path, 'rb'))
    if n_classes == 3:
        prediction['jump'] = 0
        prediction.loc[(prediction['target'] == 1) | (prediction['target'] == 2), 'jump'] = 1
    if n_classes == 2:
        prediction['jump'] = prediction.loc[:, 'target']

    test_returns = returns.loc[prediction.index, :]
    test_returns.columns = ['real_return']
    logreturns = np.log(returns + 1)
    test_logreturns = logreturns.loc[prediction.index, :]
    test_logreturns.columns = ['log_return']

    train_ret = returns.loc[:prediction.index[0], :].iloc[:-1, :]
    q = np.quantile(train_ret.abs(), 0.9)
    proba = (train_ret.abs() >= q).mean().values[0]

    print('#### Proba: ', proba)

    econ_pred = pd.read_csv(garch_path, index_col=0)
    econ_pred.index = pd.to_datetime(econ_pred.index, unit='s')

    prediction = prediction.join(econ_pred, on=None, how='left')
    prediction = prediction.join(test_returns, on=None, how='left')
    prediction = prediction.join(test_logreturns, on=None, how='left')
    prediction['lstm_prediction'] = 0
    prediction.loc[prediction['prediction_1'] + prediction['prediction_2'] >= proba, 'lstm_prediction'] = 1
    prediction.loc[:, 'volatility'] = np.exp(prediction['Sigma']) - 1
    prediction['garch_prediction'] = (prediction['volatility'] >= threshold).astype(int)

    return prediction, test_returns, test_logreturns


def get_metrics(prediction, test_returns):
    print('LSTM')
    print(metrics.classification_report(prediction['jump'], prediction['lstm_prediction']))
    print()
    print('GARCH')
    print(metrics.classification_report(prediction['jump'], prediction['garch_prediction']))

    # Confusion matrix
    garch_cm = metrics.confusion_matrix(prediction['jump'], prediction['garch_prediction'])
    lstm_cm = metrics.confusion_matrix(prediction['jump'], prediction['lstm_prediction'])

    g_precision, g_recall, g_fmeasure, support = metrics.precision_recall_fscore_support(prediction['jump'],
                                                                                         prediction['garch_prediction'])
    l_precision, l_recall, l_fmeasure, support = metrics.precision_recall_fscore_support(prediction['jump'],
                                                                                         prediction['lstm_prediction'])
    lstm_fnr = lstm_cm[1, 0] / np.sum(lstm_cm[1, :])
    garch_fnr = garch_cm[1, 0] / np.sum(garch_cm[1, :])

    result = {'prediction': prediction,
              'cm': {'lstm': lstm_cm, 'garch': garch_cm},
              'metrics': {'garch': [g_precision[-1], g_recall[-1], g_fmeasure[-1], garch_fnr],
                          'lstm': [l_precision[-1], l_recall[-1], l_fmeasure[-1], lstm_fnr]}}

    # POSITIVE
    positive = pd.DataFrame(index=test_returns.index, columns=['positive'])
    positive['returns'] = np.nan
    positive['price'] = np.nan
    positive.loc[prediction['jump'] == 1, 'price'] = prediction.loc[prediction['jump'] == 1, 'close']
    positive.loc[prediction['jump'] == 1, 'returns'] = test_returns.loc[prediction['jump'] == 1, 'real_return']
    positive_filt = positive['returns'][positive['returns'] != np.nan]
    positive_filt_price = positive['price'][positive['price'] != np.nan]

    # NEGATIVE
    negative = pd.DataFrame(index=test_returns.index, columns=['positive'])
    negative['returns'] = np.nan
    negative.loc[prediction['jump'] == 0, 'returns'] = test_returns.loc[prediction['jump'] == 0, 'real_return']
    negative_filt = negative['returns'][positive['returns'] != np.nan]
    # garch_filt_fp = garch_fp['returns'][garch_fp['returns'] != 0]

    # TRUE NEGATIVE
    lstm_tn = (prediction[prediction['lstm_prediction'] == 0]['jump'] == 0).astype(int)
    lstm_tn = pd.DataFrame(lstm_tn)
    lstm_tn.columns = ['tn']
    lstm_tn['price'] = np.nan
    lstm_tn.loc[lstm_tn.index[lstm_tn['tn'] == 1], 'price'] = prediction.loc[lstm_tn.index[lstm_tn['tn'] == 1], 'close']
    lstm_tn['returns'] = np.nan
    lstm_tn.loc[lstm_tn.index, 'returns'] = test_returns.loc[lstm_tn.index[lstm_tn['tn'] == 1], 'real_return']
    lstm_filt_tn = lstm_tn['returns'][lstm_tn['returns'] != np.nan]

    garch_tn = (prediction[prediction['garch_prediction'] == 0]['jump'] == 0).astype(int)
    garch_tn = pd.DataFrame(garch_tn)
    garch_tn.columns = ['tn']
    garch_tn['price'] = np.nan
    garch_tn.loc[garch_tn.index[garch_tn['tn'] == 1], 'price'] = prediction.loc[
        garch_tn.index[garch_tn['tn'] == 1], 'close']
    garch_tn['returns'] = np.nan
    garch_tn.loc[garch_tn.index, 'returns'] = test_returns.loc[garch_tn.index[garch_tn['tn'] == 1], 'real_return']
    garch_filt_tn = garch_tn['returns'][garch_tn['returns'] != np.nan]

    # TRUE POSITIVE
    lstm_tp = (prediction[prediction['lstm_prediction'] == 1]['jump'] == 1).astype(int)
    lstm_tp = pd.DataFrame(lstm_tp)
    lstm_tp.columns = ['tp']
    lstm_tp['price'] = np.nan
    lstm_tp.loc[lstm_tp.index[lstm_tp['tp'] == 1], 'price'] = prediction.loc[lstm_tp.index[lstm_tp['tp'] == 1], 'close']
    lstm_tp['returns'] = np.nan
    lstm_tp.loc[lstm_tp.index[lstm_tp['tp'] == 1], 'returns'] = test_returns.loc[
        lstm_tp.index[lstm_tp['tp'] == 1], 'real_return']
    lstm_filt_tp = lstm_tp['returns'][lstm_tp['returns'] != np.nan]
    lstm_filt_tp_price = lstm_tp['price'][lstm_tp['price'] != np.nan]

    garch_tp = (prediction[prediction['garch_prediction'] == 1]['jump'] == 1).astype(int)
    garch_tp = pd.DataFrame(garch_tp)
    garch_tp.columns = ['tp']
    garch_tp['price'] = np.nan
    garch_tp.loc[garch_tp.index[garch_tp['tp'] == 1], 'price'] = prediction.loc[
        garch_tp.index[garch_tp['tp'] == 1], 'close']
    garch_tp['returns'] = np.nan
    garch_tp.loc[garch_tp.index[garch_tp['tp'] == 1], 'returns'] = test_returns.loc[
        garch_tp.index[garch_tp['tp'] == 1], 'real_return']
    garch_filt_tp = garch_tp['returns'][garch_tp['returns'] != np.nan]
    garch_filt_tp_price = garch_tp['price'][garch_tp['price'] != np.nan]

    # FALSE POSITIVE
    lstm_fp = (prediction[prediction['lstm_prediction'] == 1]['jump'] == 0).astype(int)
    lstm_fp = pd.DataFrame(lstm_fp)
    lstm_fp.columns = ['fp']
    lstm_fp['price'] = np.nan
    lstm_fp.loc[lstm_fp.index[lstm_fp['fp'] == 1], 'price'] = prediction.loc[lstm_fp.index[lstm_fp['fp'] == 1], 'close']
    lstm_fp['returns'] = np.nan
    lstm_fp.loc[lstm_fp.index[lstm_fp['fp'] == 1], 'returns'] = test_returns.loc[
        lstm_fp.index[lstm_fp['fp'] == 1], 'real_return']
    lstm_filt_fp = lstm_fp['returns'][lstm_fp['returns'] != np.nan]

    garch_fp = (prediction[prediction['garch_prediction'] == 1]['jump'] == 0).astype(int)
    garch_fp = pd.DataFrame(garch_fp)
    garch_fp.columns = ['fp']
    garch_fp['price'] = np.nan
    garch_fp.loc[garch_fp.index[garch_fp['fp'] == 1], 'price'] = prediction.loc[
        garch_fp.index[garch_fp['fp'] == 1], 'close']
    garch_fp['returns'] = np.nan
    garch_fp.loc[garch_fp.index[garch_fp['fp'] == 1], 'returns'] = test_returns.loc[
        garch_fp.index[garch_fp['fp'] == 1], 'real_return']
    garch_filt_fp = garch_fp['returns'][garch_fp['returns'] != np.nan]

    # FALSE NEGATIVE
    lstm_fn = (prediction[prediction['lstm_prediction'] == 0]['jump'] == 1).astype(int)
    lstm_fn = pd.DataFrame(lstm_fn)
    lstm_fn.columns = ['fn']
    lstm_fn['price'] = np.nan
    lstm_fn.loc[lstm_fn.index[lstm_fn['fn'] == 1], 'price'] = prediction.loc[lstm_fn.index[lstm_fn['fn'] == 1], 'close']
    lstm_fn['returns'] = np.nan
    lstm_fn.loc[lstm_fn.index[lstm_fn['fn'] == 1], 'returns'] = test_returns.loc[
        lstm_fn.index[lstm_fn['fn'] == 1], 'real_return']
    lstm_filt_fn = lstm_fn['returns'][lstm_fn['returns'] != np.nan]

    garch_fn = (prediction[prediction['garch_prediction'] == 0]['jump'] == 1).astype(int)
    garch_fn = pd.DataFrame(garch_fn)
    garch_fn.columns = ['fn']
    garch_fn['price'] = np.nan
    garch_fn.loc[garch_fn.index[garch_fn['fn'] == 1], 'price'] = prediction.loc[
        garch_fn.index[garch_fn['fn'] == 1], 'close']
    garch_fn['returns'] = np.nan
    garch_fn.loc[garch_fn.index[garch_fn['fn'] == 1], 'returns'] = test_returns.loc[
        garch_fn.index[garch_fn['fn'] == 1], 'real_return']
    garch_filt_fn = garch_fn['returns'][garch_fn['returns'] != np.nan]

    result['positive'] = [positive, positive_filt]
    result['positive_price'] = [positive, positive_filt_price]
    result['negative'] = [negative, negative_filt]
    result['TN'] = {'garch': [garch_tn, garch_filt_tn], 'lstm': [lstm_tn, lstm_filt_tn]}
    result['TP'] = {'garch': [garch_tp, garch_filt_tp], 'lstm': [lstm_tp, lstm_filt_tp]}
    result['FP'] = {'garch': [garch_fp, garch_filt_fp], 'lstm': [lstm_fp, lstm_filt_fp]}
    result['FN'] = {'garch': [garch_fn, garch_filt_fn], 'lstm': [lstm_fn, lstm_filt_fn]}
    result['TP_price'] = {'garch': [garch_fn, garch_filt_tp_price], 'lstm': [lstm_fn, lstm_filt_tp_price]}

    return result


def get_cv_results(returns, dl_model, econ_model, threshold, cv_split, n_classes, model_epoch):
    results = {}
    for cv_split_i in range(cv_split):
        print(cv_split_i)
        prediction, test_returns, test_logreturns = get_pred(returns, dl_model, econ_model, threshold, cv_split_i,
                                                             model_epoch, cv_split, n_classes)
        results['cv%d' % cv_split_i] = get_metrics(prediction, test_returns)

    return results


def plot_class_results(results, type_, savefig=True, legend=True, title=None):
    fig, axes = plt.subplots(2, 5, figsize=(30, 10))
    c = 0
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]

            days = mdates.DayLocator()
            weeks = mdates.WeekdayLocator()
            dates = list(results['cv%d' % c]['prediction'].index)

            if type_ == 'TP_price':
                positive_filt = results['cv%d' % c]['positive_price'][-1]
                ax.plot(results['cv%d' % c]['prediction']['close'], color='lightgray', zorder=1)
            else:
                positive_filt = results['cv%d' % c]['positive'][-1]
                ax.plot(results['cv%d' % c]['prediction']['real_return'], color='lightgray', zorder=1)

            if type_ != 'positive':
                garch_filt = results['cv%d' % c][type_]['garch'][-1]
                lstm_filt = results['cv%d' % c][type_]['lstm'][-1]

                if type_ in ['TP', 'TP_price']:
                    ax.scatter(positive_filt.index, positive_filt, color='blue', label='positive', zorder=2, s=5)
                elif type_ == 'FN':
                    ax.scatter(positive_filt.index, positive_filt, color='blue', label='positive', zorder=2, s=5)

                ax.scatter(lstm_filt.index, lstm_filt, color='red', label='lstm %s' % type_, zorder=2, s=5)
                ax.scatter(garch_filt.index, garch_filt + 0.0005, color='green', label='garch %s' % type_, zorder=2,
                           s=5)
            else:
                ax.scatter(positive_filt.index, positive_filt, color='blue', label='positive', zorder=2, s=5)

            xfmt = mdates.DateFormatter("%y-%m-%d")
            ax.xaxis.set_major_locator(weeks)
            ax.xaxis.set_minor_locator(days)
            datemin = dates[0]
            datemax = dates[-1]
            ax.set_xlim(datemin, datemax)
            if type_ == 'TP_price':
                ax.set_ylim(2500, 12000)
            else:
                ax.set_ylim(-0.08, 0.08)

            ax.xaxis.set_major_formatter(xfmt)
            c += 1

    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper left')

    if title:
        fig.suptitle(title, fontsize=16)

    if savefig:
        plt.savefig('plot_%s.png' % type_)
