# -*- coding: utf-8 -*-
"""
Created on 9/15/18
@author: MRChou

Scenario: for converting rawdata images with labels into tfrecord files.
"""

import logging
import pickle
import pathlib

import tensorflow as tf

from TF_Utils.ImgPipeline.img_feature_proto import build_oid_feature
from TF_Utils.ImgPipeline.WriteTFRecord import ImgObjAbstract, write_tfrecord

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s',
                    handlers=[logging.FileHandler('RSNA_WriteTFR.log'),
                              logging.StreamHandler()])


with open('/archive/RSNA/ID_TO_LABELS.pkl', 'rb') as f:
    ID_TO_LABEL = pickle.load(f)


def _get_id_from_path(path):
    return pathlib.PurePath(path).stem


def _parse_labels(label):
    if label:
        return {'xmins': label['x'],
                'xmaxs': [xmin + width for xmin, width in
                          zip(label['x'], label['width'])],
                'ymins': label['y'],
                'ymaxs': [ymin + height for ymin, height in
                          zip(label['y'], label['height'])],
                'classes': [int(label['Target'])] * len(label['x'])
                }
    else:
        return {'xmins': [], 'xmaxs': [], 'ymins': [], 'ymaxs': [], 'classes': []}


class ImgObj(ImgObjAbstract):
    """
    A class that holds both image content and its labels.

    Args:
        imgid : an identifier for source image
        path  : the location of image
        label : an dictionary with feature names: value
                    {'feature_1' : feature 1 value,
                     'feature_2' : feature 2 value,
                     ...}
                   feature names must match proto defined by flag argument.
    """

    def __init__(self, imgid, path, label):
        self.imgid = imgid
        self.img_path = path
        self.labels = _parse_labels(label)

    def __str__(self):
        return 'ImgObj: imgid {}\n' \
               '        file_path {}\n' \
               '        label {}'.format(self.imgid, self.img_path, self.labels)

    def to_tfexample(self):
        tf_features = build_oid_feature(self.imgid,
                                        self.img_path,
                                        self.labels['xmins'],
                                        self.labels['xmaxs'],
                                        self.labels['ymins'],
                                        self.labels['ymaxs'],
                                        self.labels['classes']
                                        )
        return tf.train.Example(features=tf.train.Features(feature=tf_features))


if __name__ == '__main__':

    img_path = '/rawdata/RSNA_Pneumonia/imgs_train/'

    path_gener = pathlib.Path(img_path).iterdir()

    imgobj_gener = (ImgObj(imgid=_get_id_from_path(path),
                           path=str(path),
                           label=ID_TO_LABEL[_get_id_from_path(path)])
                    for path in path_gener if not path.is_dir())

    write_tfrecord(imgobj_gener=imgobj_gener,
                   num_imgs_per_file=1000,
                   fout='/archive/RSNA/trainTFRs/train.tfrecord')

    # --

    img_path = '/rawdata/RSNA_Pneumonia/imgs_test/'

    path_gener = pathlib.Path(img_path).iterdir()

    imgobj_gener = (ImgObj(imgid=_get_id_from_path(path),
                           path=str(path),
                           label=[])
                    for path in path_gener if not path.is_dir())

    write_tfrecord(imgobj_gener=imgobj_gener,
                   num_imgs_per_file=1000,
                   fout='/archive/RSNA/testTFRs/test.tfrecord')