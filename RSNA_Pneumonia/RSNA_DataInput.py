# -*- coding: utf-8 -*-
"""
Created on 10/14/18
@author: MRChou

Scenario: data input functions for estimator.

"""

import tensorflow as tf

from TF_Utils.ImgPipeline.img_feature_proto import OIDPROTO


def _parse_func(example):
    parsed_feature = tf.parse_single_example(example, features=OIDPROTO)

    imgid = parsed_feature['image/source_id']
    img = tf.image.decode_jpeg(parsed_feature['image/encoded'], channels=3)
    img = tf.cast(img, tf.float32)

    cls = parsed_feature['image/object/class/index']
    ymins = parsed_feature['image/object/bbox/ymin']
    xmins = parsed_feature['image/object/bbox/xmin']
    ymaxs = parsed_feature['image/object/bbox/ymax']
    xmaxs = parsed_feature['image/object/bbox/xmax']

    return imgid, img, cls, ymins, xmins, ymaxs, xmaxs


def _crop_left_lung(imgid, img, cls, ymins, xmins, ymaxs, xmaxs):
    img = tf.image.crop_to_bounding_box(img, 0, 0, 1024, 512)
    index = tf.where(xmins < 512)
    ymins = tf.gather_nd(ymins, index)
    ymaxs = tf.gather_nd(ymaxs, index)
    xmins = tf.gather_nd(xmins, index)
    xmaxs = tf.gather_nd(xmaxs, index)
    imgid = imgid + '_left'
    index = tf.minimum(tf.size(index), 1)
    return imgid, img, index, ymins, xmins, ymaxs, xmaxs


def _crop_right_lung(imgid, img, cls, ymins, xmins, ymaxs, xmaxs):
    img = tf.image.crop_to_bounding_box(img, 0, 512, 1024, 512)
    index = tf.where(xmins > 512)
    ymins = tf.gather_nd(ymins, index)
    ymaxs = tf.gather_nd(ymaxs, index)
    xmins = tf.gather_nd(xmins, index) - 512
    xmaxs = tf.gather_nd(xmaxs, index) - 512
    imgid = imgid + '_right'
    index = tf.minimum(tf.size(index), 1)
    return imgid, img, index, ymins, xmins, ymaxs, xmaxs


def _normalized(imgid, img, index, ymins, xmins, ymaxs, xmaxs):
    ymin = ymins[:1] / 1024
    xmin = xmins[:1] / 512
    ymax = ymaxs[:1] / 1024
    xmax = xmaxs[:1] / 512
    return imgid, img, index, ymin, xmin, ymax, xmax


def _crop_helper(*args):
    left_lung = tf.data.Dataset.from_tensors(_crop_left_lung(*args))
    right_lung = tf.data.Dataset.from_tensors(_crop_right_lung(*args))
    return left_lung.concatenate(right_lung)


def _neg_filter(imgid, img, index, ymin, xmin, ymax, xmax):
    return tf.not_equal(index, 0)


def _identity(*args):
    return args


def _vertical_flip(imgid, img, index, _ymin, xmin, _ymax, xmax):
    img = tf.image.flip_up_down(img)
    ymin = 1 - _ymax
    ymax = 1 - _ymin
    imgid = imgid + '_vertical'
    return imgid, img, index, ymin, xmin, ymax, xmax


def _horizontal_flip(imgid, img, index, ymin, _xmin, ymax, _xmax):
    img = tf.image.flip_left_right(img)
    xmin = 1 - _xmax
    xmax = 1 - _xmin
    imgid = imgid + '_horizontal'
    return imgid, img, index, ymin, xmin, ymax, xmax


def _rot180(imgid, img, index, _ymin, _xmin, _ymax, _xmax):
    img = tf.image.rot90(img, k=2)
    ymin = 1 - _ymax
    ymax = 1 - _ymin
    xmin = 1 - _xmax
    xmax = 1 - _xmin
    imgid = imgid + '_rotate'
    return imgid, img, index, ymin, xmin, ymax, xmax


def _flip_rot_helper(*args):
    vert = tf.data.Dataset.from_tensors(_vertical_flip(*args))
    rot = tf.data.Dataset.from_tensors(_rot180(*args))
    hori = tf.data.Dataset.from_tensors(_horizontal_flip(*args))

    ori = tf.data.Dataset.from_tensors(_identity(*args))
    ori = ori.concatenate(vert)
    ori = ori.concatenate(rot)
    ori = ori.concatenate(hori)
    return ori


def _horiflip_helper(*args):
    # Score difference between eval set and train set are consistent,
    # which makes me believe to turn off the verical flip/ and rotation
    hori = tf.data.Dataset.from_tensors(_horizontal_flip(*args))
    ori = tf.data.Dataset.from_tensors(_identity(*args))
    ori = ori.concatenate(hori)
    return ori


def _drop_bndbox(imgid, img, index, *args):
    return imgid, img, tf.reshape(index, [1])


def _get_bndbox(imgid, img, index, ymin, xmin, ymax, xmax):
    bbox = tf.concat([ymin, xmin, ymax, xmax], axis=0)
    return imgid, img, bbox


def _std_img(imgid, img, index, ymin, xmin, ymax, xmax):
    img = tf.image.per_image_standardization(img)
    return imgid, img, index, ymin, xmin, ymax, xmax


def keras_input_fn(files,
                   batch=1,
                   epoch=1,
                   include_neg=True,
                   augment=False,
                   stdimg=False):
    dataset = tf.data.TFRecordDataset(files,
                                      buffer_size=2048)
    dataset = dataset.map(_parse_func)
    dataset = dataset.flat_map(_crop_helper)
    dataset = dataset.map(_normalized)

    if not include_neg:
        dataset = dataset.filter(_neg_filter)

    if augment == 'horizontal':
        dataset = dataset.flat_map(_horiflip_helper)
    elif augment:
        dataset = dataset.flat_map(_flip_rot_helper)

    if stdimg:
        dataset = dataset.map(_std_img)

    if include_neg:
        dataset = dataset.map(_drop_bndbox)
    else:
        dataset = dataset.map(_get_bndbox)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(epoch)
    dataset = dataset.make_one_shot_iterator()
    imgid, img, index_or_bbox = dataset.get_next()
    return {'input_1': img, 'image_id': imgid}, index_or_bbox


if __name__ == '__main__':
    pass
