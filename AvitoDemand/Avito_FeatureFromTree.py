# -*- coding: utf-8 -*-
"""
Created on Thu June 25 07:41 2018
@author: MRChou

Code for using trees to transform features.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import os
from scipy.sparse import load_npz
from sklearn.ensemble import GradientBoostingClassifier


PKL_PATH = '/archive/Avito/data_preprocess/'
F03_TRAIN_NPZ = 'F03_train.npz'
F03_TEST_NPZ = 'F03_train.npz'
RANDOM_STATE = 42  # the random seed for sklearn train_test_split

# load data
# df = pickle.load(open(os.path.join(PKL_PATH, F02_TRAIN_PKL), 'rb'))
mat_train = load_npz(os.path.join(PKL_PATH, F03_TRAIN_NPZ))
print('Finish loading data.')

gbdt = GradientBoostingClassifier()


