import pandas as pd
import numpy as np
from constant import QS_NAME, W_NAME


def get_dl_perf_df(path):
    dl_result = pd.read_pickle('%s/dl_result.p' % path)
    qs_name = ['01', '025', '05', '10']
    ws_name = ['woneday', 'w4months', 'w6months']
    compdf = dl_result[(0.01, 24)]['df'][['returns']].copy()

    for w in ws_name:
        wdf = pd.DataFrame()
        for q in qs_name:
            usualc = '%s_0.%s' % (w, q)

            qdf = dl_result[(QS_NAME[q], W_NAME[w])]['df'].drop('returns', 1)
            qdf.columns = ['%s_%s' % (c, usualc) for c in
                           qdf.columns]  # [w + '_' + 'q0.%s_' % q + c for c in qdf.columns]
            wdf = pd.concat([wdf, qdf], 1)

            if q == '10':
                c = '%s_0.1' % w
            else:
                c = '%s_0.%s' % (w, q)

        compdf = pd.concat([compdf, wdf], 1)
    compdf = compdf.loc[:, ~compdf.columns.duplicated()]

    if np.sum(compdf.isna().sum()) != 0:
        print('Warning: There are NaNs in compdf...')
        print('Dropping NaNs...')
        compdf.dropna(inplace=True)

    return compdf
