# -*- coding: utf-8 -*-
"""
Created on 9/18/18
@author: MRChou

Scenario: scripts for creating file 'ID_TO_LABELS.pkl','LABELS_TO_CLSINDEX.pkl',
          'LBAELS_TO_DESCRIPTION.pkl'
"""

import re
import pickle

import pandas


def script_label_to_labelindex():
    file = "/rawdata/Google_OpenImg/inclusive-2018-class-descriptions.csv"
    sheet = pandas.read_csv(file)

    # there are some indices in train labels but not in file.
    # see /archive/Inclusive/sanity-check.log
    sheet = sheet.append(pandas.DataFrame([['/m/01bl7v', 'Boy'],
                                           ['/m/05r655', 'Girl'],
                                           ['/m/04yx4', 'Man'],
                                           ['/m/03bt1vf', 'Woman'],
                                           ['/m/019p5q', '?'],
                                           ['/m/02zsn', '?'],
                                           ['/m/05zppz', '?'],
                                           ['/m/02pkb8', '?']],
                                          columns=['label_code', 'description']),
                         ignore_index=True)

    label_to_index = {value: key for key, value
                      in sheet.to_dict(orient='dict')['label_code'].items()}

    label_to_des = {value['label_code']: value['description'] for value
                    in sheet.to_dict(orient='index').values()}

    with open('/archive/Inclusive/LABELS_TO_CLSINDEX.pkl', 'wb') as fout1:
        pickle.dump(label_to_index, file=fout1)

    with open('/archive/Inclusive/LABELS_TO_DESCRIPTION.pkl', 'wb') as fout2:
        pickle.dump(label_to_des, file=fout2)


def script_img_id_to_label():
    file = "/rawdata/Google_OpenImg/inclusive-2018-train_human_labels.csv"
    sheet = pandas.read_csv(file)
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
    script_label_to_labelindex()
