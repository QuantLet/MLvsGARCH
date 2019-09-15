import pickle, matplotlib, math, keras, os, time
import numpy as np

np.random.seed(7)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, TensorBoard
from sklearn import metrics
from collections import Counter
from random import shuffle
import pandas as pd
import datetime as dt

AVAILABLE_FEATURES = ['ROCP', 'log_ROCP', 'ewm_price', 'ewm_ROCP', 'ewm_log_ROCP']


# data
def load_data(path='../data/btc_1H_20160101_20190101.csv', features=None, label='labelPeakOverThreshold',
              **kwargs_label):
    """
    load raw data, build features and target
    :param path: str, path to data source
    :param features: list, features names
    :param kwargs_label: dict, parameter of labelling function
    :return: dfdata (pandas.DataFrame), target (str)
    """

    dfdata = pd.read_csv(path, index_col=0)
    dfdata.index = pd.to_datetime(dfdata.index)
    dfdata = dfdata.dropna()
    dfdata = dfdata.astype(np.float64)
    target = 'btc_usdt'

    # load label
    if label == 'labelPeakOverThreshold':
        npLabels = labelPeakOverThreshold(dfdata.close, **kwargs_label)
        labels = pd.DataFrame(npLabels,
                              columns=[target],
                              index=dfdata.index)
        dfdata['target'] = labels[target].values

    if label == 'labelQuantile':
        npLabels, label_returns, quantiles = labelQuantile(dfdata.close.values, **kwargs_label)
        # dflabel = pd.DataFrame(index = dfdata.index)
        dfdata['target'] = npLabels
        dfdata['returns'] = label_returns
        dfdata['lower'] = quantiles[:, 0]
        dfdata['upper'] = quantiles[:, 1]

    # load_features
    feature_names = []
    if features is not None:
        for feature in features:
            feature_name = feature['name']
            assert feature_name in AVAILABLE_FEATURES
            if feature_name == 'ROCP':
                dffeature = dfdata[['close']].pct_change(feature['params']['timeperiod'])

            elif feature_name == 'log_ROCP':
                dffeature = np.log(dfdata[['close']].pct_change(feature['params']['timeperiod']) + 1)

            elif feature_name == 'ewm_price':
                dffeature = dfdata[['close']].ewm(span=feature['params']['timeperiod']).mean()

            elif feature_name == 'ewm_ROCP':
                dffeature = dfdata[['close']].pct_change(feature['params']['rocp_timeperiod'])
                dffeature = dffeature.ewm(span=feature['params']['ewm_timeperiod']).mean()

            elif feature_name == 'ewm_log_ROCP':
                dffeature = np.log(dfdata[['close']].pct_change(feature['params']['rocp_timeperiod']) + 1)
                dffeature = dffeature.ewm(span=feature['params']['ewm_timeperiod']).mean()

            else:
                print('%s is not available' % feature_name)

            feature_name = feature_name + '_' + '_'.join('{}_{}'.format(*p) for p in feature['params'].items())
            dffeature.columns = [feature_name]
            feature_names.append(feature_name)

            dfdata = pd.concat([dfdata, dffeature], axis=1)

    target = 'target'

    return dfdata, target, feature_names


# label
def labelQuantile(close,
                  lq=0.1,
                  uq=0.9,
                  lookfront=1,
                  window=30,
                  threshold=None,
                  log=False,
                  fee=0,
                  binary=False):
    """

    :param close: numpy, close price
    :param lq: float, lower quantile
    :param uq: float, upper quantile
    :param lookfront: int, horizon forecast
    :param window: int, rolling window size for computing the quantile
    :param log: boolean, log scale or simple
    :param fee: float, fee
    :param binary: boolean, output is two classes or three classes
    :return:
    """

    hist_returns = np.zeros(len(close), dtype=float)

    if log:
        hist_returns[1:] = np.log(close[1:] / close[0:-1])
    else:
        hist_returns[1:] = close[1:] / close[0:-1] - 1

    labels = np.zeros(len(close), dtype=int)
    returns = np.zeros(len(close), dtype=float)

    lower_q = np.zeros(len(close), dtype=float)
    upper_q = np.zeros(len(close), dtype=float)

    for t in range(window, len(close) - lookfront):
        data_w = hist_returns[t - window: t]

        if threshold is not None:
            losses = data_w[data_w < -threshold]
            gains = data_w[data_w >= threshold]
            lower_q_t = np.quantile(losses, lq)
            upper_q_t = np.quantile(gains, uq)
        else:
            lower_q_t = np.quantile(data_w, lq)  # rolling = returns.rolling(window) q10 = rolling.quantile(0.1)
            upper_q_t = np.quantile(data_w, uq)

        for i in range(1, lookfront + 1):
            ratio = hist_returns[t + i]
            if ratio <= lower_q_t:
                if binary:
                    labels[t] = 1
                else:
                    labels[t] = 2
                break
            elif ratio >= upper_q_t:
                labels[t] = 1
                break

        returns[t] = hist_returns[t + i]
        lower_q[t] = lower_q_t
        upper_q[t] = upper_q_t

    quantiles = np.concatenate([lower_q.reshape(-1, 1),
                                upper_q.reshape(-1, 1)],
                               axis=1)

    return labels, returns, quantiles


def labelQuantileDF(close, lq=0.1, up=0.9, lookfront=1, window=30, log=False):
    """
    close: pd.dataframe
    """

    if log:
        returns = np.log(close).diff()
    else:
        returns = close.pct_change()

    lowerband = returns.expanding(window).quantile(lq)
    upperband = returns.expanding(window).quantile(up)

    quantiles = pd.concat([lowerband, upperband], axis=1)
    quantiles.columns = ['lower', 'upper']

    labels = pd.DataFrame(index=returns.index, columns=['label'])
    labels['label'] = 0

    labels.loc[(returns <= lowerband).values.reshape(-1), 'label'] = 2
    labels.loc[(returns >= upperband).values.reshape(-1), 'label'] = 1

    return labels, quantiles


def labelPeakOverThreshold(close,
                           lowerBand=-0.05,
                           upperBand=0.05,
                           lookfront=5,
                           binary=False,
                           log=False,
                           fee=0):
    labels = np.zeros(len(close), dtype=int)
    returns = np.zeros(len(close), dtype=float)
    for t in range(len(close) - lookfront):
        enter = True
        for i in range(1, lookfront + 1):
            upperBand_i = upperBand if (type(upperBand) != np.ndarray) else 1 + upperBand[t + 1]
            lowerBand_i = lowerBand if (type(lowerBand) != np.ndarray) else 1 + lowerBand[t + 1]
            ratio = close[t + i] / close[t] - 1
            if ratio <= lowerBand_i:
                if binary:
                    labels[t] = 1
                else:
                    labels[t] = 2
                break
            elif ratio >= upperBand_i and enter:
                labels[t] = 1
                break
        returns[t] = (ratio + 1) * (1 - fee) ** 2 - 1
    return labels


# Main

## Create model

class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self, optimizer='adam', loss='categorical_crossentropy', model_dir=None):
        self.loss = loss
        self.optimizer = optimizer
        if model_dir is None:
            self.model_dir = 'saved_models/%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'))

        else:
            self.model_dir = model_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.history = {}
        self.history['val_loss'] = []
        self.history['val_acc'] = []
        self.history['loss'] = []
        self.history['acc'] = []

    def load_model(self):
        print('[Model] Loading model from file %s' % self.model_dir)
        self.model = load_model(self.model_dir)

    def build_model(self, input_timesteps, input_dim, output_dim, layers):
        inputs = Input(shape=(input_timesteps, input_dim), name="input")

        for i, layer in enumerate(layers):
            if i == 0:
                network = inputs
            layer_type = layer['type']
            name = layer_type + '_' + str(i)
            if layer_type == 'LSTM':
                network = LSTM(layer['neurons'],
                               **layer['params'],
                               name=name)(network)
            elif layer_type == 'Dense':
                network = Dense(layer['neurons'],
                                **layer['params'],
                                name=name)(network)
            elif layer_type == 'softmax_output':
                outputs = Dense(output_dim, activation="softmax", name="softmax_output")(network)

        # tensor = LSTM(neurons, return_sequences=False, recurrent_dropout=0.2, name="lstm")(inputs)
        # tensor = Dense(2, activation="tanh", name="dense")(tensor)
        # outputs = Dense(3, activation="softmax", name="output")(tensor)

        self.model = keras.models.Model(
            inputs=[inputs],
            outputs=[outputs]
        )

        print(self.model.summary())

    def train_generator(self, train_generator,
                        steps_per_epoch,
                        epochs,
                        test_generator=None,
                        class_weight=None,
                        initial_epoch=0,
                        use_multiprocessing=False, validation_steps=None):

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           sample_weight_mode=None,
                           metrics=['accuracy'],
                           )
        print('[Model] Model Compiled')

        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batches per epoch' % (epochs, steps_per_epoch))
        model_dir = self.model_dir
        save_fname = model_dir

        callbacks = [
            ModelCheckpoint(filepath=save_fname + '/callback', monitor='loss', save_best_only=True)
        ]  # Early stopping ??
        print('Training class weight: ', class_weight)

        hist = self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,
            class_weight=class_weight,
            initial_epoch=initial_epoch,
            use_multiprocessing=use_multiprocessing,
            validation_data=test_generator,
            validation_steps=validation_steps
        )
        if 'val_loss' in hist.history:
            save_fname = model_dir + '/model_{}_val_loss={:.3f},val_acc={:.3f},loss={:.3f},acc={:.3f}.h5' \
                .format(
                initial_epoch + epochs - 1,
                hist.history['val_loss'][-1],
                hist.history['val_acc'][-1],
                hist.history['loss'][-1],
                hist.history['acc'][-1]
            )
            self.history['val_loss'].append(hist.history['val_loss'][-1])
            self.history['val_acc'].append(hist.history['val_acc'][-1])
            self.history['loss'].append(hist.history['loss'][-1])
            self.history['acc'].append(hist.history['acc'][-1])

        else:
            save_fname = model_dir + '/model_{}.h5' \
                .format(
                initial_epoch
            )
            self.history['loss'].append(hist.history['loss'][-1])
            self.history['acc'].append(hist.history['acc'][-1])

        self.model.save(save_fname)
        pickle.dump(self.history, open('history.p', 'wb'))
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict(self, data, input_timesteps=30, n_classes=3, loss_function=None, on_train=False):
        x_test, y_test, date_test = data.get_data(input_timesteps=input_timesteps,
                                                  n_classes=n_classes,
                                                  loss_function=loss_function,
                                                  train=on_train)

        prediction = self.model.predict(x_test)
        if loss_function == 'binary_crossentropy':
            a = np.zeros((len(y_test), 2))
            a[list(range(len(y_test))), y_test.T[0, :].astype(int)] = 1
            y_test = a
            print('prediction1', np.shape(prediction))
            prediction = np.concatenate((1 - prediction, prediction), axis=1)
            print('prediction2', np.shape(prediction))
        elif loss_function == 'categorical_crossentropy':
            print('categorical_crossentropy')
        return prediction, y_test, date_test


## data generator
class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, dfdata, features, target, seq_len, n_classes,
                 lookfront=None, class_weight=True,
                 cv_split_i=None, cv_split=1, cv_test_start=0.0,
                 normalization={'mean': None, 'sigma': None, 'n_ma': None}):
        """
        """

        print('DataLoader')
        self.seq_len = seq_len
        before_drop = len(dfdata)
        self.database = dfdata
        dffeatures = dfdata[features]
        dftarget = dfdata.loc[:, [target]]
        dfdata = pd.concat([dffeatures, dftarget], axis=1)
        dfdata.dropna(inplace=True)
        print(dfdata.head())

        assert list(dfdata.columns)[-1] == 'target', 'The last column must correspond to the target variable'

        self.feature_cols = list(dffeatures.columns)

        all_data = np.arange(0, len(dfdata))

        if cv_split is not None:
            test_start = int(len(dfdata) * cv_test_start)
            test_size = int((len(dfdata) - test_start) / cv_split)
            self.test_index = np.r_[(test_start + test_size * cv_split_i): (test_start + test_size * (cv_split_i + 1))]
            self.train_index = np.r_[0:(test_start + test_size * cv_split_i)]

        self.test_index = np.r_[self.train_index[-self.seq_len:], self.test_index]
        self.train_index = self.train_index[: -lookfront]

        # normalization with sd estimated with moving average
        mean = normalization['mean']
        sigma = normalization['sigma']
        n_ma = normalization['n_ma']
        if mean is None:
            norm_nans = True
            mean = dfdata[self.feature_cols].rolling(n_ma).mean()
        if sigma is None:
            norm_nans = True
            sigma = (dfdata[self.feature_cols] ** 2).rolling(n_ma).mean() ** 0.5

        # If we performed normalization with moving average with now have NaNs at beginning of train set
        if norm_nans:
            self.train_index = self.train_index[n_ma - 1:]

        mean = mean.dropna()
        sigma = sigma.dropna()

        self.train_index_time = dfdata.index[self.train_index]
        self.test_index_time = dfdata.index[self.test_index]
        self.dfdata_train = dfdata.iloc[self.train_index]
        self.dfdata_test = dfdata.iloc[self.test_index]
        self.len_train = len(self.dfdata_train)
        self.len_test = len(self.dfdata_test)

        self.label_train = dfdata.get(target).values[self.train_index]
        self.label_test = dfdata.get(target).values[self.test_index]

        self.len_train_windows = None

        self.class_weight = {}
        self.class_weight_count(class_weight)
        print('class_weight', self.class_weight)

        self.max_lookback = 0
        self.n_classes = n_classes

        dfdata[self.feature_cols] -= mean
        self.mean = mean
        dfdata[self.feature_cols] /= sigma
        self.sigma = sigma
        self.dfdata_train[self.feature_cols] = dfdata.iloc[self.train_index][self.feature_cols]
        self.dfdata_test[self.feature_cols] = dfdata.iloc[self.test_index][self.feature_cols]

        # creating numpy version of data to have fast slicing
        self.dfdata_train_np = self.dfdata_train.values
        self.dfdata_test_np = self.dfdata_test.values

        # print (self.dfdata_train.to_string(), file=open("da_row_data_norm.txt", "w"))
        labels, n_label = np.unique(self.label_train, return_counts=True)
        if n_classes > 0:
            print('LABELS:')
            for i in range(len(labels)):
                print('{}: {:.2f}%'.format(labels[i], n_label[i] / np.sum(n_label) * 100))
        print('Training set from %s to %s' % (list(self.dfdata_train.index.astype(str))[0],
                                              list(self.dfdata_train.index.astype(str))[-1]))

    def class_weight_count(self, class_weight=None):
        if isinstance(class_weight, (dict)):
            for k in class_weight.keys():
                self.class_weight[int(k)] = class_weight[k]
        else:
            if class_weight:
                self.class_weight = self.class_weight
            cnt = Counter(self.label_train)
            for key, value in cnt.items():
                if class_weight:
                    self.class_weight[key] = self.len_train / value
                else:
                    self.class_weight[key] = 1

    def get_data(self, input_timesteps=30, n_classes=3, loss_function=None, train=False):
        """
        Return a generator of training data from filename on given list of cols split for train/test

        :param input_timesteps: int, number of input time steps to use for one prediction
        :param n_classes: int, number of classes in the output
        :param loss_function: optional
        :param train: boolean, return training data or testing data
        :return: x_bacth, y_batch, t_batch
        """

        if train:
            len_data = self.len_train
            data = self.dfdata_train
        else:
            len_data = self.len_test
            data = self.dfdata_test

        x_batch = []
        y_batch = []
        t_batch = []
        for seq_number in range(input_timesteps, len_data):
            x = data[self.feature_cols][seq_number - input_timesteps + 1: seq_number + 1: 1].values
            x_batch.append(x)
            y = data['target'].iloc[[seq_number]].values
            if n_classes > 0:
                y_hot_encoded = np.zeros(n_classes)
                y_hot_encoded[int(y)] = 1
                y_batch.append(y_hot_encoded)
            else:
                y_batch.append(y)

            t_batch.append(data.index[seq_number])

        x_batch = np.array(x_batch)

        y_batch = np.array(y_batch)
        if loss_function == 'binary_crossentropy':
            y_batch = np.array([y_batch[:, 1]]).T

        return x_batch, y_batch, t_batch

    def generate_train_batch_random(self, batch_size, train=True,
                                    input_timesteps=30, loss_function=None, n_classes=3):
        """Yield a generator of training data from filename on given list of cols split for train/test

        params
        ------

        """

        i = 0

        self.shuffled_mapping = list(range(input_timesteps + self.max_lookback - 1, self.len_train, 1))
        shuffle(self.shuffled_mapping)

        while True:
            start = time.time()
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                seq_number = self.shuffled_mapping[i]
                x = self.dfdata_train_np[seq_number - input_timesteps + 1: seq_number + 1: 1, :-1]
                x_batch.append(x)
                y = self.dfdata_train_np[seq_number, -1]

                if n_classes > 0:
                    y_hot_encoded = np.zeros(n_classes)
                    y_hot_encoded[int(y)] = 1
                    y_batch.append(y_hot_encoded)
                else:
                    y_batch.append(y)

                i += 1
                if i >= len(self.shuffled_mapping):
                    i = 0

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            if loss_function == 'binary_crossentropy':
                y_batch = np.array([y_batch[:, 1]]).T

            yield x_batch, y_batch

    def generate_test_batch_random(self, batch_size, train=True,
                                   input_timesteps=30, loss_function=None, n_classes=3):
        """Yield a generator of training data from filename on given list of cols split for train/test

        params
        ------

        """

        i = 0

        self.shuffled_mapping = list(range(input_timesteps + self.max_lookback - 1, self.len_train, 1))
        shuffle(self.shuffled_mapping)

        while True:
            start = time.time()
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                seq_number = self.shuffled_mapping[i]

                x = self.dfdata_test_np[seq_number - input_timesteps + 1: seq_number + 1: 1, :-1]

                x_batch.append(x)

                y = self.dfdata_test_np[seq_number, -1]

                if n_classes > 0:
                    y_hot_encoded = np.zeros(n_classes)
                    y_hot_encoded[int(y)] = 1
                    y_batch.append(y_hot_encoded)
                else:
                    y_batch.append(y)

                i += 1
                if i >= len(self.shuffled_mapping):
                    i = 0

            x_batch = np.array(x_batch)

            y_batch = np.array(y_batch)
            if loss_function == 'binary_crossentropy':
                y_batch = np.array([y_batch[:, 1]]).T

            yield x_batch, y_batch


def plot_roc_curve(y_test, preds, figure_dir, fig_name, create_plot=True, legend=True, color='b', plot_type='roc',
                   class_i=1, label_append=''):
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=1)

    roc_auc = metrics.auc(fpr, tpr)

    plt.rcParams["figure.figsize"] = (10, 10)
    if plot_type == 'roc':
        plt.plot(fpr, tpr, color=color, label='AUC = %0.2f ' % roc_auc + 'for {}'.format(class_i) + label_append)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if legend:
        plt.legend(loc='lower right')
    if create_plot:
        plt.savefig('{}/{}-ROC.png'.format(figure_dir, fig_name),
                    bbox_inches='tight')
        plt.clf()


def plot_performance(target, y_test, predictions, date_test, figure_dir, fig_name,
                     data_loader, classification=True):
    results = data_loader.database.loc[date_test, [target]]
    results.columns = ['target']

    if classification:
        n_classes = np.size(y_test, 1)
        for class_i in range(n_classes):
            results['prediction_{}'.format(class_i)] = predictions[:, class_i]
    else:
        results['prediction'] = predictions

    for key in ['high', 'close', 'open', 'low']:
        results[key] = data_loader.database.loc[date_test, [key]]

    results.to_pickle('{}/{}-prediction.pkl'.format(figure_dir, fig_name))

    if classification:
        if n_classes == 3:
            for class_i in range(3):
                y_test_i = y_test + 0
                y_test_i[:, 0] = np.max(y_test[:, [k for k in range(n_classes) if k != class_i]], axis=1)
                y_test_i[:, 1] = y_test[:, class_i]
                y_test_i = y_test_i[:, 0:2]
                predictions_i = predictions + 0
                predictions_i[:, 0] = np.sum(predictions[:, [k for k in range(n_classes) if k != class_i]], axis=1)
                predictions_i[:, 1] = predictions[:, class_i]
                predictions_i = predictions_i[:, 0:2]

                plot_roc_curve(np.asarray(y_test_i).T[1, :], predictions_i[:, 1],
                               figure_dir, fig_name + '_one_vs_all', create_plot=(class_i == (n_classes - 1)),
                               color=['b', 'r', 'g'][class_i], class_i=class_i)

            for class_i in range(3):
                class_0 = (class_i + 1) % 3
                class_1 = class_i
                y_test_i = y_test + 0
                y_test_i[:, 0] = y_test[:, class_0]
                y_test_i[:, 1] = y_test[:, class_1]
                y_test_i = y_test_i[:, 0:2]
                predictions_i = predictions + 0
                predictions_i[:, 0] = predictions[:, class_0]
                predictions_i[:, 1] = predictions[:, class_1]
                predictions_i = predictions_i[:, 0:2]
                I = np.sum(y_test_i, axis=1) != 0
                y_test_i = y_test_i[I, :]
                predictions_i = predictions_i[I, :]
                plot_roc_curve(np.asarray(y_test_i).T[1, :], predictions_i[:, 1], figure_dir, fig_name + '_one_vs_one',
                               create_plot=(class_i == (n_classes - 1)),
                               color=['b', 'r', 'g'][class_0], class_i=class_0, label_append=str(class_1))


def train_predict(dfdata, target,  # data_path='../data/btc_1H_20160101_20190101.csv',
                  model=None,
                  model_dir=None,
                  features=['ROCP'],
                  cv_split_i=0,
                  cv_split=1,
                  cv_test_start=0.7,
                  batch_size=50,
                  initial_epoch=0,
                  epochs=10,
                  loss_function='categorical_crossentropy',
                  class_weight=True,
                  input_timesteps=30,
                  n_classes=3,
                  lookfront=1,
                  normalization={'mean': None, 'sigma': None, 'n_ma': 100},
                  layers=None,
                  training=True):
    """
    main function to train a lstm neural network on one training set
    and predict on corresponding validation set.

    """
    if model is None:
        # Create model
        m_dir = model_dir + '/' + str(cv_split_i)
        model = Model(optimizer='adam', loss=loss_function, model_dir=m_dir)
        model.build_model(input_timesteps, len(features), n_classes, layers)

    # Create data generator for cross validation
    data = DataLoader(dfdata, features, target, input_timesteps, n_classes,
                      lookfront=lookfront, class_weight=class_weight,
                      cv_split_i=cv_split_i, cv_split=cv_split, cv_test_start=cv_test_start,
                      normalization=normalization)

    if class_weight:
        cweights = data.class_weight
    else:
        cweights = None
    # main training loop
    if training:
        steps_per_epoch = math.ceil(data.len_train / batch_size)
        # validation_steps = math.ceil(data.len_test / batch_size) WRONG
        model.train_generator(
            data.generate_train_batch_random(
                batch_size=batch_size,
                input_timesteps=input_timesteps,
                loss_function=loss_function,
                n_classes=n_classes,
                train=True
            ),
            steps_per_epoch=steps_per_epoch,
            validation_steps=None,
            test_generator=None,
            epochs=initial_epoch + epochs,
            initial_epoch=initial_epoch,
            use_multiprocessing=False,
            class_weight=cweights
        )
        # Predict on corresponding validation set

        predictions, y, date = model.predict(data,
                                             input_timesteps=input_timesteps,
                                             n_classes=n_classes,
                                             loss_function=loss_function)

    else:
        # Predict on corresponding training set
        predictions, y, date = model.predict(data,
                                             input_timesteps=input_timesteps,
                                             n_classes=n_classes,
                                             loss_function=loss_function,
                                             on_train=True)

    return predictions, model, y, data, date, target


# model selection methods
def roc_curve(y_test, preds, figure_dir, fig_name, create_plot=True, legend=True, color='b', plot_type='roc', class_i=1,
              label_append=''):
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # plt.title('Receiver Operating Characteristic')
    if plot_type == 'roc':
        plt.plot(fpr, tpr, color=color, label='AUC = %0.2f ' % roc_auc + 'for {}'.format(class_i) + label_append)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if legend:
        plt.legend(loc='lower right')

    if create_plot:
        plt.savefig('{}/{}-ROC.svg'.format(figure_dir, fig_name))
        plt.clf()


def plot_selection(dir_, plot_type):
    cv = [int(d) for d in os.listdir(dir_) if d.isdigit()]
    cv.sort()
    n_epochs = int(list(filter(lambda x: 'prediction' in x, os.listdir(dir_ + '/' + str(cv[0]))))[0].split('-')[1])
    cv_k = len(cv)

    plt.figure(figsize=(n_epochs * 6 * 1.5, (cv_k + 1) * 3 * 1.5))

    for epoch_number in range(n_epochs):
        print(epoch_number)
        for cv_split_i in cv:
            pred_dir = '{}/{}/e={}-{}-prediction.pkl'.format(dir_, str(cv_split_i), epoch_number, n_epochs)
            data_i = pickle.load(open(pred_dir, 'rb'))

            plt.subplot(cv_k + 1, n_epochs, 1 + epoch_number + cv_split_i * n_epochs)

            if plot_type == 'roc_one_vs_one':
                n_classes = 3
                predictions = data_i[['prediction_{}'.format(i) for i in range(n_classes)]].values
                y_test = data_i['target']
                y_test = y_test.values
                y_test_0 = np.zeros((len(y_test), n_classes))
                y_test_0[list(range(len(y_test))), y_test] = 1
                y_test = y_test_0
                for class_i in range(3):
                    class_0 = (class_i + 1) % 3
                    class_1 = class_i
                    y_test_i = y_test + 0
                    y_test_i[:, 0] = y_test[:, class_0]
                    y_test_i[:, 1] = y_test[:, class_1]
                    y_test_i = y_test_i[:, 0:2]
                    predictions_i = predictions + 0
                    predictions_i[:, 0] = predictions[:, class_0]
                    predictions_i[:, 1] = predictions[:, class_1]
                    predictions_i = predictions_i[:, 0:2]
                    I = np.sum(y_test_i, axis=1) != 0
                    y_test_i = y_test_i[I, :]
                    predictions_i = predictions_i[I, :]
                    roc_curve(np.asarray(y_test_i).T[1, :], predictions_i[:, 1], None, 'all', create_plot=False,
                              color=['b', 'r', 'g'][class_0], class_i=class_0, label_append=str(class_1))

    save_dir = dir_ + '/selection/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig('{}{}.png'.format(save_dir, plot_type))


def get_total_pred(dir_='./saved_models/12072019-143851/', epoch_number='9', file_name='total'):
    cv = [int(d) for d in os.listdir(dir_) if d.isdigit()]
    cv.sort()
    n_epochs = int(list(filter(lambda x: 'prediction' in x, os.listdir(dir_ + '/' + str(cv[0]))))[0].split('-')[1])

    total_pred = pd.DataFrame()

    for cv_split_i in cv:
        pred_dir = '{}/{}/e={}-{}-prediction.pkl'.format(dir_, str(cv_split_i), epoch_number, n_epochs)
        pred = pickle.load(open(pred_dir, 'rb'))
        total_pred = pd.concat([total_pred, pred])

    if file_name:
        total_pred.to_pickle('{}/e={}-{}-{}_prediction.pkl'.format(dir_, epoch_number, n_epochs, file_name))

    return total_pred


def get_total_roc_curve(dir_='./saved_models/12072019-143851/', epoch_number='9', fig_name='Total', legend=False):
    '''cv = [int(d) for d in os.listdir(dir_) if d.isdigit()]

    cv.sort()
    n_epochs = int(list(filter(lambda x: 'prediction' in x, os.listdir(dir_ + '/' + str(cv[0]))))[0].split('-')[1])

    total_pred = pd.DataFrame()

    for cv_split_i in cv:
        pred_dir = '{}/{}/e={}-{}-prediction.pkl'.format(dir_, str(cv_split_i), epoch_number, n_epochs)
        pred = pickle.load(open(pred_dir, 'rb'))
        total_pred = pd.concat([total_pred, pred])
    pickle.dump(total_pred, '{}.e={}-{}-total_prediction.pkl'.format(dir_, epoch_number, n_epochs))
    '''

    total_pred = get_total_pred(dir_, epoch_number)
    predictions = total_pred[['prediction_0', 'prediction_1', 'prediction_2']].values
    label = total_pred['target'].values

    n_classes = len(np.unique(label))
    y_test = np.zeros((len(label), n_classes))
    y_test[np.arange(len(label)), label] = 1

    if n_classes == 3:
        for class_i in range(3):
            y_test_i = y_test + 0
            y_test_i[:, 0] = np.max(y_test[:, [k for k in range(n_classes) if k != class_i]], axis=1)
            y_test_i[:, 1] = y_test[:, class_i]
            y_test_i = y_test_i[:, 0:2]
            predictions_i = predictions + 0
            predictions_i[:, 0] = np.sum(predictions[:, [k for k in range(n_classes) if k != class_i]], axis=1)
            predictions_i[:, 1] = predictions[:, class_i]
            predictions_i = predictions_i[:, 0:2]

            plot_roc_curve(np.asarray(y_test_i).T[1, :], predictions_i[:, 1],
                           dir_, fig_name + '_one_vs_all', create_plot=(class_i == (n_classes - 1)),
                           color=['b', 'r', 'g'][class_i], class_i=class_i)

        for class_i in range(3):
            class_0 = (class_i + 1) % 3
            class_1 = class_i
            y_test_i = y_test + 0
            y_test_i[:, 0] = y_test[:, class_0]
            y_test_i[:, 1] = y_test[:, class_1]
            y_test_i = y_test_i[:, 0:2]
            predictions_i = predictions + 0
            predictions_i[:, 0] = predictions[:, class_0]
            predictions_i[:, 1] = predictions[:, class_1]
            predictions_i = predictions_i[:, 0:2]
            I = np.sum(y_test_i, axis=1) != 0
            y_test_i = y_test_i[I, :]
            predictions_i = predictions_i[I, :]
            plot_roc_curve(np.asarray(y_test_i).T[1, :], predictions_i[:, 1], dir_, fig_name + '_one_vs_one',
                           create_plot=(class_i == (n_classes - 1)),
                           color=['b', 'r', 'g'][class_0], class_i=class_0, label_append=str(class_1),
                           legend=legend)
