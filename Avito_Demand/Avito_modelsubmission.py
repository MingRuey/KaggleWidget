# -*- coding: utf-8 -*-
"""
Created on Thu May. 23. 2018
@author: MRChou

Code used for create submission file for models.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""
import os
import pickle
import numpy
import pandas
import xlearn
import xgboost as xgb
from scipy.sparse import load_npz
from keras.models import load_model
from Avito_TrainScripts import feature_scaling


def ffm_submit(model, test, file_name='submission.txt'):
    ffm_model = xlearn.create_ffm()
    ffm_model.setTest(test)
    ffm_model.predict(model, './' + file_name)


def xgb_submit(model, df_test, file_name='submission.csv'):
    """Create the submission file of the model on test data."""
    xgb_test = xgb.DMatrix(
        data=df_test.drop(['item_id', 'description'], axis=1)
        )

    df_test['deal_probability'] = model.predict(xgb_test)

    # if prediction is < 0, let it be 0
    df_test.loc[df_test['deal_probability'] < 0, 'deal_probability'] = 0
    df_test.loc[df_test['deal_probability'] > 1, 'deal_probability'] = 1
    df_test[['item_id', 'deal_probability']].to_csv(file_name, index=False)
    return None


def lgbm_submit(model, df_test, file_name='submission.csv'):
    """Create the submission file of the model on test data."""

    df_test['deal_probability'] = model.predict(
        df_test.drop(['item_id', 'description'], axis=1))

    # if prediction is < 0, let it be 0
    df_test.loc[df_test['deal_probability'] < 0, 'deal_probability'] = 0
    df_test.loc[df_test['deal_probability'] > 1, 'deal_probability'] = 1
    df_test[['item_id', 'deal_probability']].to_csv(file_name, index=False)
    return None


def lgbm_submmit_sparse(model,
                        sparse_mat,
                        item_id,
                        file_name='submission.csv'):
    """lgbm_submit with data in scipy sparse matrix."""

    # make prediction
    df_test = pandas.DataFrame({'deal_probability': model.predict(sparse_mat)})

    # add item_id to df_test
    df_test['item_id'] = item_id

    # if prediction is < 0, let it be 0
    df_test.loc[df_test['deal_probability'] < 0, 'deal_probability'] = 0
    df_test.loc[df_test['deal_probability'] > 1, 'deal_probability'] = 1
    df_test[['item_id', 'deal_probability']].to_csv(file_name, index=False)
    return None


def nn_submit(model, df_test, file_name='submission.csv'):
    model = load_model(model)

    feature_scaling(df_test)

    df_test['deal_probability'] = model.predict(
        df_test.drop(['item_id', 'description'], axis=1)
    )

    # if prediction is < 0, let it be 0; if is > 1, let it be 1
    df_test.loc[df_test['deal_probability'] < 0, 'deal_probability'] = 0
    df_test.loc[df_test['deal_probability'] > 1, 'deal_probability'] = 1
    df_test[['item_id', 'deal_probability']].to_csv(file_name, index=False)
    return None


def script_ffm():
    path = '/archive/Avito/data_preprocess/'

    test = '/archive/Avito/data_preprocess/FFM_test.txt'
    model = './av03_ffm_0607.txt'
    ffm_submit(model, test)

    with open(os.path.join(path, 'test.pickle'), 'rb') as f:
        predict = numpy.loadtxt('./av03_ffm_0607_output.txt')
        df_test = pickle.load(f)
        df_test['deal_probability'] = predict
        df_test[['item_id', 'deal_probability']].to_csv(
            'av03_ffm_0607_submission.csv', index=False)


def script_xgb():
    path = '/archive/Avito/data_preprocess/'

    with open('/archive/Avito/models/av06_xgb_0611.pickle', 'rb') as fin1:
        with open(os.path.join(path, 'F02_test.pickle'), 'rb') as fin2:
            model = pickle.load(fin1)
            print('Evaluation on vali: ', model.best_score)

            df_test = pickle.load(fin2)
            xgb_submit(model, df_test, file_name='av06_xgb_submission.csv')


def script_lgbm():
    path = '/archive/Avito/data_preprocess/'

    with open('/archive/Avito/models/av08_lgb-gbdt_0627.pickle', 'rb') as fin1:
        with open(os.path.join(path, 'F02_test.pickle'), 'rb') as fin2:
            model = pickle.load(fin1)
            print('Evaluation on vali: ', model.best_score)

            # df_test = pickle.load(fin2)
            # lgbm_submit(model, df_test, file_name='av07_lgbm_submission.csv')
            item_ids = pickle.load(fin2)['item_id'].values
            mat_test = load_npz(os.path.join(path, 'F04_test.npz'))
            lgbm_submmit_sparse(model,
                                mat_test,
                                item_ids,
                                file_name='av08_lgb-gbdt_submission.csv'
                                )


def script_nn():
    path = '/archive/Avito/data_preprocess/'

    with open(os.path.join(path, 'F01_test.pickle'), 'rb') as f:
        model = '/home/mrchou/code/KaggleWidget/av01_nn.h5'
        df_test = pickle.load(f)
        nn_submit(model, df_test, file_name='av01_nn_submission.csv')


def blend_model():
    path = '/archive/Avito/submissions/'
    file_name = 'ensemble_publics_submission.csv'

    model_av07 = pandas.read_csv(path + 'av07_lgb-gbdt_submission.csv')
    model_public0 = pandas.read_csv(path + 'public0_02211.csv')
    model_public1 = pandas.read_csv(path + 'public1_02212.csv')
    model_public2 = pandas.read_csv(path + 'public2_02212.csv')
    model_public3 = pandas.read_csv(path + 'public3_02237.csv')
    model_public4 = pandas.read_csv(path + 'public4_02246.csv')
    model_public5 = pandas.read_csv(path + 'public5_02211.csv')

    models = [model_public0, model_public1, model_public2,
              model_public3, model_public4, model_public5]
    weights = [0.2211, 0.2212, 0.2212, 0.2237, 0.2246, 0.2211]

    # initialize dataframe
    result = pandas.DataFrame()
    result['item_id'] = model_av07['item_id']
    result['deal_probability'] = 0.

    # normalize the weights
    weights = [i/sum(weights) for i in weights]

    for model, weight in zip(models, weights):
        result['deal_probability'] += model['deal_probability']*weight

    # if prediction is < 0, let it be 0
    result.loc[result['deal_probability'] < 0, 'deal_probability'] = 0
    result.loc[result['deal_probability'] > 1, 'deal_probability'] = 1
    result[['item_id', 'deal_probability']].to_csv(file_name, index=False)


if __name__ == '__main__':
    blend_model()
