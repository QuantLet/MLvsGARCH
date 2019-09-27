import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics

def proba_to_class(prediction, proba, class_ = 2):
    pred = pd.DataFrame(index = prediction.index, columns = ['pred'])
    pred['target'] = 0
    pred.loc[:, 'pred'] = 0
    if class_ == 1:
        pred.loc[prediction['target'] == 1, 'target'] = 1
        pred.loc[prediction['target'] == 2, 'target'] = 0
        pred.loc[prediction['prediction_1'] >= proba, 'pred'] = 1
    elif class_ == 2:
        pred.loc[prediction['target'] == 1, 'target'] = 0
        pred.loc[prediction['target'] == 2, 'target'] = 1
        pred.loc[prediction['prediction_2'] >= proba, 'pred'] = 1
        
    return pred

def get_optimal_threshold_new(target, prediction, probas, level, type_ = 'f_measure'):
    results = np.zeros((100,4))
    for i,p in enumerate(probas):
        pred = (prediction >= p).astype(int)
        cm = metrics.confusion_matrix(target, pred, labels = [0,1])
        fn = cm[1,0]
        tp = cm[1,1]
        fp = cm[0,1]
        tn = cm[0,0]
        
        results[i,0] = p
        results[i,1] = fn
        results[i,2] = tp
        #results[i,3] = fp/(tn+fp) # fn/len(prediction)
        results[i,3] = fn/len(prediction)
        #results[i,3] = fn/(tp+fn) # FNR
        #results[i,3] = fn/(tn + fn)
        
        
    results = results[results[:,3].argsort()]
    proba_level = results[results[:,3] <= level, 0]
    #print(proba_level)
    
    results = np.zeros((100,4))
    for i,p in enumerate(proba_level):
        
        pred = (prediction >= p).astype(int)
        cm = metrics.confusion_matrix(target, pred, labels = [0,1])
        fn = cm[1,0]
        tp = cm[1,1]
        fp = cm[0,1]
        tn = cm[0,0]
         
        results[i,0] = p
        results[i,1] = fn
        results[i,2] = tp
        
        if type_ == 'fnr':
            results[i,3] = 1 - fn/(tp+fn) # 1 - FNR = TPR # type II error
        elif type_ == 'fpr':
            results[i,3] = 1 - fp/(tn+fp) # FPR  # type I error
        elif type_ == 'f_measure':
            results[i,3] = metrics.f1_score(target, pred, labels = [0,1])
        elif type_ == 'precision':
            results[i,3] = metrics.precision_score(target, pred, labels = [0,1])
        elif type_ == 'recall':
            results[i, 3] = metrics.recall_score(target, pred, labels = [0,1])
        
    # sort in ascending order given metrics
    results = results[results[:,3].argsort()]
    proba = results[:, 0][-1]
    
        
    return {'test': results, 'proba': proba}

def get_optimal_t_from_auc(target=None, pred=None):
        fpr, tpr, t = metrics.roc_curve(target, pred, pos_label=1)
        #auc = metrics.roc_auc_score(target, pred)
        random_classifier = np.linspace(0, 1, len(tpr))
        dist = (tpr - random_classifier)
        max_ = dist.argmax()
        t_opt = t[max_]

        return t_opt

def get_optimal_t_cost(target=None, proba_class=None, returns = None):
    print('Computing cost matrix from target')
    avg_ret_0 = returns.loc[target.index[target == 0]].mean()
    avg_ret_1 = returns.loc[target.index[target == 1]].mean()
    cost_TN = 0
    cost_FP = - avg_ret_0
    cost_FN = 0
    cost_TP = - avg_ret_1

    cost_matrix = np.array([[cost_TN, cost_FP],
                            [cost_FN, cost_TP]])

    threshold = np.linspace(0, 1, 200)
    scores = np.zeros(len(threshold))
    for i,t in enumerate(threshold):
        pred_1 = (proba_class > t).astype(int)
        cm = metrics.confusion_matrix(target, pred_1)
        scores[i] = np.sum(cost_matrix * cm)
    return threshold[scores.argmax()]

def get_optimal_proba(func, target, prediction, probas = None, level = None, type_ = None, returns = None):
    if func == 'get_optimal_t_from_auc':
        optimal_t = get_optimal_t_from_auc(target, prediction)
    elif func == 'get_optimal_threshold_new':
        results = get_optimal_threshold_new(target, prediction, probas, level, type_)
        optimal_t = results['proba']
    elif func == 'get_optimal_t_cost':
        optimal_t = get_optimal_t_cost(target, prediction, returns)
    return optimal_t
    
def cv_optimal_proba(func, target, prediction, window, every, probas = None, level = None, type_ = None, save_path = None, returns = None):

    optimal_prob = np.ones(len(prediction), dtype=float) * 1000
    cv_i = 0
    for i in range(window, len(prediction)):
        if i % every == 0:
            start = i - window - cv_i * every
            pred = prediction.iloc[start:i]
            tar_i = target.iloc[start:i]
            
            optimal_prob[i] = get_optimal_proba(func, 
                                                tar_i, 
                                                pred, 
                                                probas = probas, 
                                                level = level, 
                                                type_ = type_,
                                                returns = returns)
            print(optimal_prob[i])
            cv_i += 1
            
    if save_path is not None:
        pickle.dump(optimal_prob, open('%s_w%s_every%s_level%s_optimal_prob.csv' % (save_path, window, every, level), 'wb'))
            
    return optimal_prob

def build_binary_target(dl_pred):
    dl_pred['pred_01'] = dl_pred.loc[:, 'prediction_0'] + dl_pred.loc[:, 'prediction_1']
    dl_pred['pred_02'] = dl_pred.loc[:, 'prediction_0'] + dl_pred.loc[:, 'prediction_2']
    dl_pred['pred_12'] = dl_pred.loc[:, 'prediction_1'] + dl_pred.loc[:, 'prediction_2']

    dl_pred['target_drop'] = 0
    dl_pred.loc[dl_pred['target'] == 1, 'target_drop'] = 0
    dl_pred.loc[dl_pred['target'] == 2, 'target_drop'] = 1

    dl_pred['target_jump'] = 0
    dl_pred.loc[dl_pred['target'] == 2, 'target_jump'] = 0
    
    return dl_pred


def get_pred_from_train(dir_, epoch_number, n_epochs, window,
                        func = None, probas = None, level = None, type_ = None ,returns = None):

    list_dir_ = os.listdir(dir_)
    cvs = np.array(list_dir_)[[d.isdigit() for d in list_dir_]]
    cvs = cvs.astype(int)
    cvs.sort()
    print(cvs)

    train_prediction = pd.DataFrame()
    prediction = pd.DataFrame()

    for cv_i in cvs:
        print('CV %s' % cv_i)
        # Train pred
        train_pred = pickle.load(open('{}/{}/e={}-{}_train-prediction.pkl'.format(dir_, cv_i, epoch_number, n_epochs), 
                                      'rb')
                                )
        val_pred =  pickle.load(open('{}/{}/e={}-{}-prediction.pkl'.format(dir_, cv_i, epoch_number, n_epochs), 
                                      'rb')
                                )
        train_pred = build_binary_target(train_pred)
        train_pred = train_pred.iloc[-window:,:]
        val_pred = build_binary_target(val_pred)

        optimal_p = get_optimal_proba(func,
                                      train_pred['target_drop'],
                                      train_pred['prediction_2'],
                                      probas = probas,
                                      level = level,
                                      type_ = type_,
                                      returns = returns)
        print('Optimal proba: ', optimal_p)

        # Train pred
        train_pred['pred_drop'] = 0
        train_pred.loc[train_pred['prediction_2'] >= optimal_p, 'pred_drop'] = 1
        # Validation
        val_pred['pred_drop'] = 0
        val_pred.loc[val_pred['prediction_2'] >= optimal_p, 'pred_drop'] = 1
        val_pred['probas'] = optimal_p


        train_prediction = pd.concat([train_prediction, train_pred])
        prediction = pd.concat([prediction, val_pred])
        
    return train_prediction, prediction