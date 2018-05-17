# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:32:19 2018
@author: MRChou

Include tools for processing large csv files. 
Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import pandas

PATH = 'Y:/Avito_Demand/'

df = pandas.read_csv(PATH+'train_active.csv', nrows=500)
print(df)