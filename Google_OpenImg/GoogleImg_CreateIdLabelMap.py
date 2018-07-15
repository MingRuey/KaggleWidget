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

from CnnUtils import BBox

OUTFILE = 'ImgId_to_BboxLabels.pkl'
ANNO = '/rawdata/Google_OpenImg/challenge-2018-train-annotations-bbox.csv'

IdLabelMap = defaultdict(list)
for index, row in pandas.read_csv(ANNO).iterrows():
    bbox = BBox(*row[['LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values)
    IdLabelMap[row['ImageID']].append(bbox)

with open(OUTFILE, 'wb') as fout:
    pickle.dump(IdLabelMap, fout)
