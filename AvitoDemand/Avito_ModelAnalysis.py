# -*- coding: utf-8 -*-
"""
Created on Thu June 11 10:00 2018
@author: MRChou

Code used for analysis the models in Avito competition.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import os
import pickle
from math import floor
import lightgbm
from sklearn.model_selection import train_test_split

MODEL_PATH = '/archive/Avito/models/'
PKL_PATH = '/archive/Avito/data_preprocess/'
AV04 = 'av04_lgb-gbdt_0608.pickle'
F02_TRAIN_PKL = 'F02_train.pickle'
RANDOM_STATE = 42  # the random seed for sklearn train_test_split


def lgbm_geterror(lgb_model, data, label, measure='mse'):
    error = lgb_model.predict(data) - label
    if measure == 'mse':
        return error**2


def script_lgbm_highesterror(n):
    with open(MODEL_PATH + AV04, 'rb') as f_model:
        with open(PKL_PATH + F02_TRAIN_PKL, 'rb') as f_data:
            model = pickle.load(f_model)
            df = pickle.load(f_data)

            df['error'] = lgbm_geterror(model,
                                        df.drop(['item_id',
                                                 'description',
                                                 'deal_probability'],
                                                axis=1).values,
                                        df['deal_probability'].values
                                        )

            df.sort_values(by=['error'], inplace=True)

            if isinstance(n, int):
                return df.head(n)
            elif isinstance(n, float):
                assert 0 < n < 1
                return df.head(floor(n*df.shape[0]))
            else:
                raise ValueError('n must be int of float.')


if __name__ == '__main__':
    pass

