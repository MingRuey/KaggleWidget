# -*- coding: utf-8 -*-
"""
Created on July 13 19:30 2018
@author: MRChou

Create a dict to map image id into it's bounding box labels.
Store the dict into a pickle file.
"""

import pickle
from collections import defaultdict

import pandas

from CnnUtils.ImgObj import BBox

OUTANNO = 'ImgId_to_BboxLabels.pkl'
OUTCLSID = 'LabelName_to_ClassID.pkl'
ANNO = '/rawdata/Google_OpenImg/challenge-2018-train-annotations-bbox.csv'
CLS = '/rawdata/Google_OpenImg/challenge-2018-class-descriptions-500.csv'

IdLabelMap = defaultdict(list)
for index, row in pandas.read_csv(ANNO).iterrows():
    bbox = BBox(*row[['LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values)
    IdLabelMap[row['ImageID']].append(bbox)

with open(ANNO, 'wb') as fout:
    pickle.dump(IdLabelMap, fout)


# print out message when finding duplicate labels
label_duplicate_msg = 'LabelName %s duplicate with index = %s and index = %s'

# generate actual mapping as a dictionary then store into a pickle file.
ClsMap = dict()
table = pandas.read_csv(CLS, header=None, names=['LabelName', 'LabelContent'])
for index, row in table.iterrows():
    if row['LabelName'] not in ClsMap:
        ClsMap[row['LabelName']] = index+1
    else:
        print(label_duplicate_msg % (row['LabelName'],
                                     index,
                                     ClsMap[row['LabelName']])
              )

with open(OUTCLSID, 'wb') as fout:
    pickle.dump(ClsMap, fout)
