# -*- coding: utf-8 -*-
"""
Created on 9/17/18
@author: MRChou

Scenario: scripts for creating file 'ID_TO_LABELS.pkl'
"""

import pickle

import numpy
import pandas


def script():
    sheet = pandas.read_csv('/rawdata/RSNA_Pneumonia/stage_1_train_labels.csv')

    id_field = 1  # patientId
    label_fields = {2: 'x', 3: 'y', 4: 'width', 5: 'height'}
    target = 6
    id_to_labels = {}

    def getlabels(row):
        labels = {label_fields[field]: [row[field]] for field in label_fields}
        labels['Target'] = row[target]
        return labels

    def process_row(row):
        labels = getlabels(row)
        patientId = row[id_field]

        # rationale check: Target value 0 should not have labels, vice versa
        for field in label_fields.values():
            if (pandas.isna(labels[field])) != (labels['Target'] == 0):
                if labels['Target'] == 0:
                    msg = 'Patient {} with normal Target and non-Nan label {}.'
                    print(msg.format(patientId, field))
                else:
                    msg = 'Patient {} with Target {} and Nan label {}.'
                    print(msg.format(patientId,
                                     labels['Target'],
                                     field))

        # if patient already exists
        if patientId in id_to_labels:
            old_labels = id_to_labels[patientId]

            # rationale check: Target match
            if not labels['Target'] == old_labels['Target']:
                msg = 'Patient {} has inconsistent Target {} and {}.'
                print(msg.format(patientId,
                                 labels['Target'],
                                 old_labels['Target']))

            # rationale check 2: label should be different
            elif all([labels[field] == old_labels[field] for field in label_fields.values()]):
                msg = 'Patient {} has duplicate records.'
                print(msg.format(patientId))

            else:
                for field in label_fields.values():
                    labels[field] += old_labels[field]

        id_to_labels[patientId] = labels

    for row in sheet.itertuples():
        process_row(row)

    with open('/archive/RSNA/ID_TO_LABELS.pkl', 'wb') as fout:
        pickle.dump(id_to_labels, file=fout)


if __name__ == '__main__':
    script()
