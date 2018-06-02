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
from keras.models import load_model
from Avito_TrainScripts import feature_scaling


def lgbm_submit(model, df_test):
    """Create the submission file of the model on test data."""
    df_test['deal_probability'] = model.predict(
        df_test.drop(['item_id', 'description'], axis=1))

    # if prediction is < 0, let it be 0
    df_test.loc[df_test['deal_probability'] < 0, 'deal_probability'] = 0
    df_test[['item_id', 'deal_probability']].to_csv(
        'lgb_gbdt_0522_Av00_submission.csv', index=False)
    return None


def nn_submit(model, df_test):
    model = load_model(model)

    feature_scaling(df_test)

    df_test['deal_probability'] = model.predict(
        df_test.drop(['item_id', 'description'], axis=1)
    )

    # if prediction is < 0, let it be 0; if is > 1, let it be 1
    df_test.loc[df_test['deal_probability'] < 0, 'deal_probability'] = 0
    df_test.loc[df_test['deal_probability'] > 1, 'deal_probability'] = 1
    df_test[['item_id', 'deal_probability']].to_csv(
        'av01_nn_submission.csv', index=False)
    return None


def cnn_submit():
    pass


def script_lgbm():
    path = '/archive/Avito/data_preprocess/'

    with open('/archive/Avito/models/lgb-gbdt_0522_Av00.pickle', 'rb') as fin1:
        with open(os.path.join(path, 'test_F01.pickle'), 'rb') as fin2:
            model = pickle.load(fin1)
            df_test = pickle.load(fin2)
            lgbm_submit(model, df_test)


def script_nn():
    path = '/archive/Avito/data_preprocess/'

    with open(os.path.join(path, 'F01_test.pickle'), 'rb') as f:
        model = '/home/mrchou/code/KaggleWidget/av01_nn.h5'
        df_test = pickle.load(f)
        nn_submit(model, df_test)


if __name__ == '__main__':
    script_nn()
