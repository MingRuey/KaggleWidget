# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:32:19 2018
@author: MRChou

Code used for fast lightBGM model building.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import os
import pickle
import lightgbm
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

TRAIN_PATH = '/rawdata/Avito_Demand/imgs_train/'
PKL_PATH = '/archive/Avito/data_preprocess/'
TRAIN_PKL = 'train.pickle'
F04_TRAIN = 'F04_train.npz'
F04_TEST = 'F04_test.npz'
LABELS = 'F04_ColumnInfo.pickle'
RANDOM_STATE = 42  # the random seed for sklearn train_test_split

# load data
# df = pickle.load(open(os.path.join(PKL_PATH, F02_TRAIN_PKL), 'rb'))
mat_train = load_npz(os.path.join(PKL_PATH, F04_TRAIN))
print('Finish loading data.')

# random split train-vali
train, vali = train_test_split(mat_train,
                               test_size=0.2,
                               random_state=RANDOM_STATE
                               )

# create train/vali data
# lgb_train = lightgbm.Dataset(data=train.drop(['item_id',
#                                               'description',
#                                               'deal_probability'],
#                                              axis=1),
#                              label=train['deal_probability']
#                              )
#
# lgb_vali = lightgbm.Dataset(data=vali.drop(['item_id',
#                                             'description',
#                                             'deal_probability'],
#                                            axis=1),
#                             label=vali['deal_probability'],
#                             reference=lgb_train
#                             )
with open(os.path.join(PKL_PATH, LABELS), 'rb') as f_label:
    labels = pickle.load(f_label)
    labels = labels['F05_train_columns'][:-1]

lgb_train = lightgbm.Dataset(data=train[:, :-1],
                             feature_name=labels,
                             label=train[:, -1].toarray().reshape(-1,)
                             )
lgb_vali = lightgbm.Dataset(data=vali[:, :-1],
                            feature_name=labels,
                            label=vali[:, -1].toarray().reshape(-1,)
                            )
print('Finish building Dataset for lightgbm.')

vali_rmse = float('inf')
model = None
best_leaf = 0
best_depth = 0
for leaf in [300]:
    for depth in [20]:

        # train parameters
        params = {'device': 'gpu',
                  'boosting_type': 'gbdt',
                  'objective': 'poisson',
                  'poission_max_delta_step': 1.2,
                  'metric': 'rmse',
                  'num_leaves': leaf,
                  'learning_rate': 0.1,
                  # 'max_depth': depth,
                  'verbose': -1,
                  }

        # start training
        eval_results = {}
        gbm = lightgbm.train(params=params,
                             num_boost_round=4000,
                             train_set=lgb_train,
                             valid_sets=lgb_vali,
                             evals_result=eval_results,
                             early_stopping_rounds=50,
                             verbose_eval=50
                             )

        if min(eval_results['valid_0']['rmse']) < vali_rmse:
            print('Vali loss from {} to {}, with leaf: {} and depth: {}'.format(
                  vali_rmse, min(eval_results['valid_0']['rmse']), leaf, depth)
                  )
            vali_l2 = min(eval_results['valid_0']['rmse'])
            model = gbm
            best_depth = depth
            best_leaf = leaf

            # save model with pickle
            with open('av09_lgb-gbdt_0628.pickle', 'wb') as fout:
                pickle.dump(model, fout)
                print('Model saved, leaf: {}, depth: {} and vali_l2: {}'.format(
                    best_leaf, best_depth, vali_l2)
                )
