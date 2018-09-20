# -*- coding: utf-8 -*-
"""
Created on 9/17/18
@author: MRChou

Scenario: script for sanity check on Google Inclusive Image Challenge labels.
"""

import pathlib
from itertools import islice
from pandas import read_csv, DataFrame

folder = pathlib.Path('/rawdata/Google_OpenImg/')
prefix = '/rawdata/Google_OpenImg/inclusive-2018-'

train_ids = {path.stem for path in
             (folder / 'imgs_train/').iterdir() if path.is_file()}
test_ids = {path.stem for path in
            (folder / 'inclusive-2018-imgs_test').iterdir() if path.is_file()}

sample_submit = read_csv(prefix + 'stage_1_sample_submission.csv')
labels_tune = read_csv(prefix + 'tuning_labels.csv',
                       header=None,
                       names=['ImageID', 'labels'])

labels_human = read_csv(prefix + 'train_human_labels.csv')
labels_machine = read_csv(prefix + 'train_machine_labels.csv')

cls_descript = read_csv(prefix + 'class-descriptions.csv')
cls_trainable = read_csv(prefix + 'classes-trainable.csv')

cls_ids = set(cls_descript['label_code'])
cls_train = set(cls_trainable['label_code'])
cls_human = set(labels_human['LabelName'])
cls_machine = set(labels_machine['LabelName'])


if __name__ == '__main__':
    print('Listing columns of csvs:')
    local_vars = locals().copy()
    for varname, var in local_vars.items():
        if isinstance(var, DataFrame):
            print('Columns of {}: {}'.format(varname, [col for col in var]))

    def _is_same(set1, set2):
        symm_diff = set(set1) ^ set(set2)
        return symm_diff == set(), len(symm_diff)

    def _check_subset(set1, set2):
        set1, set2 = set(set1), set(set2)
        if set1 < set2:
            return True, len(set2 - set1)
        else:
            return False, len(set1 - set2)

    print('\nAbout image ids check: \n')
    print('Train has {} IDs, Some random train IDs: {}'.format(
        len(train_ids), list(islice(train_ids, 5))))
    print('Test has {} IDs, Some random train IDs: {}'.format(
        len(test_ids), list(islice(test_ids, 5))))

    # sanity check 1-0 every test image is in sample_submit, vice versa
    #              1-1 every tune label is in test image id.
    print('sample_submit {} rows, are its images equal to test id? {}'.format(
        sample_submit.shape[0], _is_same(sample_submit['image_id'], test_ids)))
    print('tuning labels {} rows, are its images in test ids? {}'.format(
        labels_tune.shape[0], _check_subset(labels_tune['ImageID'], test_ids)))

    # sanity check 2-0 human labeled images equals train IDs, vice versa
    #              2-1 machine labeled images equals train IDs, vice versa
    print('human labels {} rows, images equal to train ids? {}'.format(
        labels_human.shape[0], _is_same(labels_human['ImageID'], train_ids)))
    print('machine labels {} rows, images euqual to train ids? {}'.format(
        labels_machine.shape[0], _is_same(labels_machine['ImageID'], train_ids)))
    print('machine-labeled images subset of  train images? {}'.format(
        _check_subset(labels_machine['ImageID'], train_ids)))

    # ---

    print('\nAbout labels check: \n')
    print('Description has {} rows, with {} ids, some samples: {}'.format(
        cls_descript.shape[0], len(cls_ids), list(islice(cls_ids, 5))))
    print('Trainable Class has {} rows, with {} ids, some samples: {}'.format(
        cls_trainable.shape[0], len(cls_train), list(islice(cls_train, 5))))

    # sanity check 3 every labels in human, machine & tune are in cls_descript
    print('Human labels has {} kind, is subset of class-descript? {}.'.format(
        len(cls_human), _check_subset(cls_human, cls_ids)))
    print('Machine labels has {} kind, is subset of class-descript? {}.'.format(
        len(cls_machine), _check_subset(cls_machine, cls_ids)))
    print('Labels by human {} but no description'.format(
        cls_human - cls_ids))
    print('Labels by machine {} but no description'.format(
        cls_machine - cls_ids))

