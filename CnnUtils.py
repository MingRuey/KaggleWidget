# -*- coding: utf-8 -*-
"""
Created on July 12 22:30 2018
@author: MRChou

Useful classes and functions for trainning CNN models (in Tensorflow).
With the intension to make it incorporate with Google object API.

Check out:
https://github.com/tensorflow/models/tree/master/research/object_detection

"""

from collections import namedtuple

import numpy
import tensorflow as tf


def _tffeature_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _tffeature_int64list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _tffeature_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tffeature_byteslist(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _tffeature_floatlist(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Image:
    """An object stores a single image."""

    def __init__(self, imgid, img_array, imgformat='jpg'):
        self._imgid = str(imgid)
        self._value = numpy.array(img_array)
        self._height, self._width, *kw = self.value.shape
        self.format = imgformat

    @property
    def imgid(self):
        return self._imgid

    @property
    def value(self):
        return self._value

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def tffeatures(self):
        imgid_in_bytes = self.imgid.encode()
        imgarr_in_bytes = self.value.tostring()
        format_in_byte = self.format.encode()

        return {'image/source_id': _tffeature_bytes(imgid_in_bytes),
                'image/filename': _tffeature_bytes(imgid_in_bytes),
                'image/format': _tffeature_bytes(format_in_byte),
                'image/height': _tffeature_int64(self.height),
                'image/width': _tffeature_int64(self.width),
                'image/encoded': _tffeature_bytes(imgarr_in_bytes)
                }


_bbox = namedtuple('BoundingBox', ['labelname', 'xmin', 'xmax', 'ymin', 'ymax'])


class BBox(_bbox):
    """Each bounding box is a namedtuple whose attributes are converted."""

    def __new__(cls, *args):
        labelname, xmin, xmax, ymin, ymax = args
        self = super(BBox, cls).__new__(cls,
                                        labelname=str(labelname),
                                        xmin=float(xmin),
                                        xmax=float(xmax),
                                        ymin=float(ymin),
                                        ymax=float(ymax)
                                        )

        assert all(0 <= index <= 1 for index in {self.xmin,
                                                 self.xmax,
                                                 self.ymin,
                                                 self.ymax}
                   )
        assert self.xmax > self.xmin and self.ymax > self.ymin

        return self


class ObjDetectImg(Image):
    """An image with bound box as labels"""

    def __init__(self, imgid, img_array, labels):
        super(ObjDetectImg, self).__init__(imgid, img_array)
        self.labels = [BBox(*label) for label in labels]

    def tffeatures(self):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        cls_txt = []
        for label in self.labels:
            xmins.append(label.xmin)
            xmaxs.append(label.xmax)
            ymins.append(label.ymin)
            ymaxs.append(label.ymax)
            cls_txt.append(label.labelname.encode())

        box_features = {'image/object/bbox/xmin': _tffeature_floatlist(xmins),
                        'image/object/bbox/xmax': _tffeature_floatlist(xmaxs),
                        'image/object/bbox/ymin': _tffeature_floatlist(ymins),
                        'image/object/bbox/ymax': _tffeature_floatlist(ymaxs),
                        'image/object/class/text': _tffeature_byteslist(cls_txt)
                        }
        return {**box_features, **super(ObjDetectImg, self).tffeatures()}


if __name__ == '__main__':
    pass
