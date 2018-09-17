# -*- coding: utf-8 -*-
"""
Created on 9/18/18
@author: MRChou

Scenario: scripts for creating file 'ID_TO_LABELS.pkl'
"""

import re
import pickle

import pandas


def script():
    sheet = pandas.read_csv("/rawdata/Google_OpenImg/inclusive-2018-train_human_labels.csv")
    id_field = 1
    label_field = 3

    id_to_labels = {}
    for row in sheet.itertuples():
        img_id, cls = row[id_field], row[label_field]

        pattern = '/m/[a-zA-Z0-9]{1,}'
        assert re.match(pattern, cls), 'Wrong format of class: {}'.format(cls)

        oid_labels = id_to_labels.get(img_id, None)
        if oid_labels is None:
            id_to_labels[img_id] = [cls]
        else:
            assert not(cls in oid_labels), 'Duplcated labels: {}'.format(img_id)
            oid_labels.append(cls)

    with open('/archive/Inclusive/ID_TO_LABELS.pkl', 'wb') as fout:
        pickle.dump(id_to_labels, file=fout)


if __name__ == '__main__':
    script()
