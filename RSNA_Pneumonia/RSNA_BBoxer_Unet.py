# -*- coding: utf-8 -*-
"""
Created on 10/15/18
@author: MRChou

Scenario: Trying out a brute force method with tf hub pretrained model.
"""

import os
import pathlib
from functools import partial

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from skimage import measure

from RSNA_Pneumonia.RSNA_DataInput import keras_input_fn
from TF_Utils.Models.UNet import unet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def unet_input_fn(*args, **kwargs):
    feature, label = keras_input_fn(*args, **kwargs)
    label = label * tf.constant([1024., 512., 1024., 512.])
    return {'image': feature['input_1'], 'image_id': feature['image_id'], 'bbox': label}, label


def unet_loss(seg_map, label):
    # TODO: Only supports sinlge batch
    assert_op = tf.Assert(tf.equal(tf.shape(label)[0], 1), [label])
    with tf.control_dependencies([assert_op]):
        bbox = tf.cast(label, tf.int32)

        # region inside bounding box:
        ymin = tf.clip_by_value(bbox[0, 0], 0, 1024)
        xmin = tf.clip_by_value(bbox[0, 1], 0, 512)
        ymax = tf.clip_by_value(bbox[0, 2], 0, 1024)
        xmax = tf.clip_by_value(bbox[0, 3], 0, 512)
        target = seg_map[:, ymin:ymax, xmin:xmax, :]
        seg_size = tf.cast(tf.size(seg_map), tf.float32)
        target_size = tf.cast(tf.size(target), tf.float32)

        # loss for target:
        target_loss = -1 * tf.log(target + 1e-7) / target_size
        target_loss = tf.reduce_sum(target_loss)

        # loss for background
        bg_loss = -1 * tf.log(1 - seg_map + 1e-7)
        target_gain = -1 * tf.log(1 - target + 1e-7)
        bg_loss = (tf.reduce_sum(bg_loss) - tf.reduce_sum(target_gain))
        bg_loss = bg_loss / (seg_size - target_size)

        return bg_loss + target_loss


def model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    # load unet
    with tf.variable_scope('UNet'):
        output = unet(features['image'], training=istrain, reg=0.001)

    if istrain or iseval:
        iou_loss = unet_loss(output, labels)
        l2_reg_loss = tf.losses.get_regularization_loss()
        loss = iou_loss + l2_reg_loss

        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, global_step=globalstep)

        tf.summary.scalar('iou_loss', iou_loss)

    else:
        loss = None
        train_op = None

    if ispredict or iseval:  # i.e. not in train mode
        predictions = {'output': output,
                       'image_id': features['image_id'],
                       'bbox': features['bbox']}
    else:
        predictions = None

    eval_metric = None
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def imgmask_to_bbox(img, threshold):
    connected = numpy.where(img > threshold, 1, 0)
    connected = measure.label(connected)

    max_area = 0
    ymin, xmin, ymax, xmax = -1, -1, -1, -1
    for region in measure.regionprops(connected):
        if region.area > max_area:
            ymin, xmin, ymax, xmax = region.bbox
            max_area = region.area
    return numpy.array([ymin, xmin, ymax, xmax])


def numpy_iou(bboxes1, bboxes2, epsilon=1e-10):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    x11, y11, x12, y12 = numpy.split(bboxes1, 4, axis=-1)
    x21, y21, x22, y22 = numpy.split(bboxes2, 4, axis=-1)

    xI1 = numpy.maximum(x11, numpy.transpose(x21))
    xI2 = numpy.minimum(x12, numpy.transpose(x22))

    yI1 = numpy.maximum(y11, numpy.transpose(y21))
    yI2 = numpy.minimum(y12, numpy.transpose(y22))

    inter_area = numpy.maximum((xI2 - xI1), 0) * numpy.maximum((yI2 - yI1), 0)

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + numpy.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union+epsilon)


def script_unet_test():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/BBoxer_Unet/'

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(unet_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            stdimg=True,
                            include_neg=False,
                            augment=False)

    lr = 0.04
    momentum = 0.5
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                   config=config,
                                   params={'sgdlr': lr, 'sgdmomt': momentum})

    for thres in [0.0, 0.25, 0.5, 0.75]:
        iou_sum = []
        for result in model.predict(input_fn=eval_input_fn):
            img = result['output']
            bbox = result['bbox']
            predict_box = imgmask_to_bbox(img, threshold=thres)
            iou = numpy_iou(bbox, predict_box)
            iou_sum.append(numpy.max(iou, axis=-1))
        print('Avg IoU: ', numpy.mean(iou_sum))
        print('Std IoU: ', numpy.std(iou_sum))
        print('Max IoU: ', numpy.max(iou_sum))
        print('Min IoU: ', numpy.min(iou_sum))


def script_train_unet():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/BBoxer_Unet_aug/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(unet_input_fn,
                             files=train_files,
                             epoch=1, batch=1,
                             include_neg=False,
                             stdimg=True,
                             augment=True)

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(unet_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            include_neg=False,
                            stdimg=True,
                            augment=False)

    lr = 0.02
    momentum = 0.5
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    for step in range(10):
        model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                       config=config,
                                       params={'sgdlr': lr,
                                               'sgdmomt': momentum})

        print(model.evaluate(input_fn=eval_input_fn))
        for thres in [0.25, 0.5, 0.75, 0.9]:
            iou_sum = []
            for result in model.predict(input_fn=eval_input_fn):
                img = result['output']
                bbox = result['bbox']
                predict_box = imgmask_to_bbox(img, threshold=thres)
                iou = numpy_iou(bbox, predict_box)
                iou_sum.append(numpy.max(iou, axis=-1))
            print('Avg IoU: ', numpy.mean(iou_sum))
            print('Std IoU: ', numpy.std(iou_sum))
            print('Max IoU: ', numpy.max(iou_sum))
            print('Min IoU: ', numpy.min(iou_sum))

        model.train(input_fn=train_input_fn)
        tf.reset_default_graph()
        lr = lr*0.95


if __name__ == '__main__':
    # script_unet_test()
    script_train_unet()
