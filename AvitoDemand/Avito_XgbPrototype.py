# -*- coding: utf-8 -*-
"""
Created on Thu June 11 03:33 2018
@author: MRChou

Code used for fast Xgboost model building.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split

TRAIN_PATH = '/rawdata/Avito_Demand/imgs_train/'
PKL_PATH = '/archive/Avito/data_preprocess/'
TRAIN_PKL = 'train.pickle'
F02_TRAIN_PKL = 'F02_train.pickle'
F02_TEST_PKL = 'F02_test.pickle'
RANDOM_STATE = 42  # the random seed for sklearn train_test_split

# load data
df = pickle.load(open(os.path.join(PKL_PATH, F02_TRAIN_PKL), 'rb'))

# random split train-vali
train, vali = train_test_split(df,
                               test_size=0.2,
                               random_state=RANDOM_STATE
                               )


# create train/vali data
xgb_train = xgb.DMatrix(data=train.drop(['item_id',
                                         'description',
                                         'deal_probability'],
                                        axis=1),
                        label=train['deal_probability']
                        )

xgb_vali = xgb.DMatrix(data=vali.drop(['item_id',
                                       'description',
                                       'deal_probability'],
                                      axis=1),
                       label=vali['deal_probability']
                       )

# train parameters
params = {'booster': 'gbtree',
          'tree_method': 'gpu_hist',
          'n_gpus': 2,
          'objective': 'gpu:reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 15,
          'eta': 0.3
          }

# start training
eval_results = {}
xgbtree = xgb.train(params=params,
                    dtrain=xgb_train,
                    num_boost_round=5000,
                    evals=[(xgb_train, 'train'), (xgb_vali, 'vali')],
                    early_stopping_rounds=50,
                    evals_result=eval_results
                    )

with open('test.pickle', 'wb') as fout:
    pickle.dump(xgbtree, fout)



