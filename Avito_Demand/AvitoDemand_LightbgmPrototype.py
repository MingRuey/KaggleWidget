# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:32:19 2018
@author: MRChou

Code used for fast lightBGM model building.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import os
import numpy
import pandas
import pickle
import lightgbm
from sklearn.metrics import mean_squared_error

PATH = '/archive/Avito/data_preprocess/'

# load data
df = pickle.load(open(os.path.join(PATH, 'train_F01.pickle'), 'rb'))

# the train-validation split
msk = numpy.random.rand(df.shape[0]) < 0.8 # a random 80-20 split
train = df[msk]
vali = df[~msk]

# create train/vali data
lgb_train = lightgbm.Dataset(data=train.drop(['item_id', 'description', 'deal_probability'], axis=1),
                             label=train['deal_probability']
                             )

lgb_vali = lightgbm.Dataset(data=vali.drop(['item_id', 'description', 'deal_probability'], axis=1),
                            label=vali['deal_probability'],
                            reference=lgb_train
                            )

# train parameters
params = {'device': 'gpu',
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mse',
          'num_leaves': 1000,
          'learning_rate': 0.005,
          'max_depth': 50,
          }

# start training
eval_results = {}
gbm = lightgbm.train(params=params,
                     num_boost_round=5000,
                     train_set=lgb_train,
                     valid_sets=lgb_vali,
                     evals_result=eval_results,
                     verbose_eval=50
                     )

# save model with pickle
with open('lgb-gbdt_0522_Av00.pickle', 'wb') as fout:
    pickle.dump(gbm, fout)

