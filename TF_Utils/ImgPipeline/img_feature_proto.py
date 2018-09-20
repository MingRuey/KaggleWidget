# -*- coding: utf-8 -*-
"""
Created on 9/15/18
@author: MRChou

Scenario: store the proto -- required features of an image, for various format.

Usage:
    CLSPROTO / OIDPROTO --- defines how to parse the resulted tfrecord file.
    build_[***]_features(*args) --- turn *args into tf.train.Example proto.
                                    Currently [***] may be 'oid' or 'cls'.
"""

from pathlib import PurePath
from collections import UserDict

import numpy
import cv2
from pydicom import dcmread
import tensorflow as tf

# proto for classification on image
CLSPROTO = {'image/source_id': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/class/index':
                tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            }

# proto for object detection on image
OIDPROTO = {'image/source_id': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmin':
                tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/bbox/xmax':
                tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/bbox/ymin':
                tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/bbox/ymax':
                tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/class/index':
                tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            }


def _tffeature_int64(value):
    value = [value] if isinstance(value, int) else value
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _tffeature_float(value):
    value = [value] if isinstance(value, float) else value
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _tffeature_bytes(value):
    if isinstance(value, str):
        value = value.encode()
    value = [value] if isinstance(value, bytes) else value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _to_jpgbyte_jpgarray(img_path):
    """Read image file and return jpg bytes string and numpy array

    Args:
        img_path: the location of image file

    Return: A tuple -- ('jpg', jpg bytes string, numpy array)
    """
    filetype = str(PurePath(img_path).suffix.strip('.'))
    if filetype == 'jpg':
        with open(img_path, 'rb') as f_img:
            img_bytes = f_img.read()
            img_array = cv2.imdecode(numpy.frombuffer(img_bytes, numpy.uint8),
                                     cv2.IMREAD_COLOR)
    elif filetype == 'dcm':
        _, img_array = cv2.imencode('.jpg', dcmread(img_path).pixel_array)
        img_bytes = img_array.tobytes()
        img_array = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    else:
        msg = 'Not supported file format: {}'.format(filetype)
        raise NotImplementedError(msg)
    return 'jpg', img_bytes, img_array


class _FeatureDict(UserDict):

    def __init__(self, flag='cls'):
        if flag == 'cls':
            self._proto = CLSPROTO
        elif flag == 'oid':
            self._proto = OIDPROTO
        else:
            raise NotImplementedError('Not supported flag: {}'.format(flag))
        initial_data = {keys: None for keys in self._proto}
        super(__class__, self).__init__(initial_data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, value):
        # check value consistent with self.proto,
        # TODO: Check on whether features are fixed length(now check only dtype)
        if item in self.data:
            try:
                if self._proto[item].dtype == tf.string:
                    self.data[item] = _tffeature_bytes(value)
                elif self._proto[item].dtype == tf.int64:
                    self.data[item] = _tffeature_int64(value)
                else:
                    self.data[item] = _tffeature_float(value)
            except TypeError as err:
                msg = 'Field {} not match proto: can not turn {} into tffeature'
                raise TypeError(msg.format(item, value)) from err
        else:
            super(__class__, self).__setitem__(item, value)


def build_cls_feature(imgid, path, classes):

    img_format, img_bytes, img_array = _to_jpgbyte_jpgarray(path)

    tf_features = _FeatureDict(flag='cls')
    tf_features.update({'image/source_id': imgid,
                        'image/filename': PurePath(path).name,
                        'image/format': img_format,
                        'image/encoded': img_bytes,
                        'image/height': img_array.shape[0],
                        'image/width': img_array.shape[1],
                        'image/class/index': list(classes)
                        })

    return {**tf_features}  # return normal dict instead _FeatureDict


def build_oid_feature(imgid, path, xmins, xmaxs, ymins, ymaxs, classes):

    img_format, img_bytes, img_array = _to_jpgbyte_jpgarray(path)

    tf_features = _FeatureDict(flag='oid')
    tf_features.update({'image/source_id': imgid,
                        'image/filename': PurePath(path).name,
                        'image/format': img_format,
                        'image/encoded': img_bytes,
                        'image/height': img_array.shape[0],
                        'image/width': img_array.shape[1],
                        'image/object/bbox/xmin': list(xmins),
                        'image/object/bbox/xmax': list(xmaxs),
                        'image/object/bbox/ymin': list(ymins),
                        'image/object/bbox/ymax': list(ymaxs),
                        'image/object/class/index': list(classes)
                        })

    return {**tf_features}  # return normal dict instead _FeatureDict


if __name__ == '__main__':
    pass
