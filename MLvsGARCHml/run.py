from core import Model, plot_performance, load_data, train_predict, get_total_roc_curve
import json, os
import datetime as dt
from keras import backend as keras_backend


def run(config,
        classification=True,
        training=True):
    data_param, label_param, training_param, cv_param, layers = config["data_param"], config["label_param"], \
                                                                     config["training_param"], config["cv_param"], \
                                                                     config["model"]

    if training:
        epoch_count = 0
        model_dir = 'saved_models/{}-{}'.format(dt.datetime.now().strftime('%d%m%Y-%H%M%S'), config["comments"])

    else:
        model_dir = 'saved_models/{}'.format(config["load_model"]["path"])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    json.dump(config, open('{}/config.json'.format(model_dir), 'w'))

    # load feature and label
    dfdata, target, feature_names = load_data(path=data_param['data_path'], features=data_param['features'],
                                              label=config['label'], **label_param)
    print(dfdata.head())

    if config['label'] == 'labelQuantile':
        dfdata.to_pickle('%s/dfdata.p' % model_dir)

    global_dates = {}

    if training:
        for cv_split_i in range(cv_param['cv_split']):
            model = None
            keras_backend.clear_session()
            for epoch_number in range(training_param['n_epochs']):
                print('Epoch %d' % epoch_number)
                predictions, model, y_test, data_loader, date_test, target = train_predict(
                    dfdata,
                    target,
                    model=model,
                    model_dir=model_dir,
                    features=feature_names,
                    cv_split_i=cv_split_i,
                    cv_split=cv_param['cv_split'],
                    cv_test_start=cv_param['cv_test_start'],
                    batch_size=training_param['batch_size'],
                    initial_epoch=epoch_number,
                    epochs=1,
                    loss_function=training_param['loss_function'],
                    class_weight=training_param['class_weight'],
                    input_timesteps=training_param['input_timesteps'],
                    n_classes=data_param['n_classes'],
                    lookfront=label_param['lookfront'],
                    normalization=data_param['normalization'],
                    layers=layers,
                    training=training)

                epoch_count += 1
                fig_name = 'e={}-{}'.format(epoch_number, training_param['n_epochs'])
                try:
                    plot_performance(target, y_test, predictions, date_test, model.model_dir, fig_name,
                                     data_loader, classification)
                except:
                    continue

            global_dates['cv_%d' % cv_split_i] = {'train': list(data_loader.train_index_time.astype(str)),
                                                  'test': list(data_loader.test_index_time.astype(str)),
                                                  'date_test': list(map(str, date_test))
                                                  }

        for epoch_number in range(training_param['n_epochs']):
            get_total_roc_curve(dir_=model_dir,
                                epoch_number=epoch_number,
                                fig_name='e_%s_total' % epoch_number,
                                legend=True)
        json.dump(global_dates, open('%s/global_dates.json' % model_dir, 'w'))

    else:
        for cv_split_i in range(cv_param['cv_split']):
            keras_backend.clear_session()
            epoch_number = config['load_model']['epoch_number']
            model = Model(model_dir='{}/{}/model_{}.h5'.format(model_dir, cv_split_i, epoch_number))
            model.load_model()

            print('Epoch %d' % epoch_number)
            predictions, model, y_test, data_loader, date_test, target = train_predict(
                dfdata,
                target,
                model=model,
                model_dir=model_dir,
                features=feature_names,
                cv_split_i=cv_split_i,
                cv_split=cv_param['cv_split'],
                cv_test_start=cv_param['cv_test_start'],
                batch_size=training_param['batch_size'],
                initial_epoch=epoch_number,
                epochs=1,
                loss_function=training_param['loss_function'],
                class_weight=training_param['class_weight'],
                input_timesteps=training_param['input_timesteps'],
                n_classes=data_param['n_classes'],
                lookfront=label_param['lookfront'],
                normalization=data_param['normalization'],
                layers=layers,
                training=training)

            fig_name = 'e={}-{}_train'.format(epoch_number, training_param['n_epochs'])
            path = '{}/{}/'.format(model_dir, cv_split_i)
            plot_performance(target, y_test, predictions, date_test, path, fig_name,
                             data_loader, classification)
            print(date_test[-1])
