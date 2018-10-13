# -*- coding: utf-8 -*-
"""
Created on 10/11/18
@author: MRChou

Scenario: Script for trainning faster rcnn.
"""

import pathlib
from functools import partial

import numpy
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from TF_Utils.Models.Resnet import ResnetV2
from TF_Utils.Models.FasterRCNN.RPN import RPN
from TF_Utils.Models.FasterRCNN.FRCNN import FastRCNN
from TF_Utils.ImgPipeline.img_feature_proto import OIDPROTO

_ORI_SHAPE = (1024, 1024)
_IMG_SHAPE = (800, 800)

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def _parse_func(example):
    parsed_feature = tf.parse_single_example(example, features=OIDPROTO)

    imgid = parsed_feature['image/source_id']

    # decode img and change dtype to float32
    img = parsed_feature['image/encoded']
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, size=_IMG_SHAPE)
    img = tf.cast(img, dtype=tf.float32)

    cls = parsed_feature['image/object/class/index']
    ymins = parsed_feature['image/object/bbox/ymin']
    xmins = parsed_feature['image/object/bbox/xmin']
    ymaxs = parsed_feature['image/object/bbox/ymax']
    xmaxs = parsed_feature['image/object/bbox/xmax']

    # create bbox
    bbox = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)
    bbox = tf.cast(bbox, tf.float32)
    bbox = bbox * _IMG_SHAPE[0]/_ORI_SHAPE[0]
    return imgid, img, cls, bbox


def _filter_func(_id, _img, cls, _bbox):
    return tf.not_equal(cls[0], 0)


def input_fn(files, epoch=1, batch=1):
    with tf.name_scope('imput_pipleline'):
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_func)
        dataset = dataset.filter(_filter_func)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        dataset = dataset.make_one_shot_iterator()
        imgid, img, cls, bbox = dataset.get_next()

        # remove batch dimension
        bbox = bbox[0, ...]
        cls = cls[0, ...]

    return {'image': img, 'image_id': imgid}, {'class': cls, 'bbox': bbox}


def _script_examine_input():
    folder = '/archive/RSNA/train_TFRs/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    with tf.Session() as sess:
        feature, label = input_fn(train_files)
        for _ in range(10):
            img, lbl = sess.run([feature, label])
            if numpy.count_nonzero(numpy.isnan(lbl['bbox'])) != 0:
                print(lbl['bbox'], lbl['class'])


def model_fn(features, labels, mode, params):
    # set global step
    globalstep = tf.train.get_or_create_global_step()

    # detect mode
    istrain = (mode == tf.estimator.ModeKeys.TRAIN)
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    inputs = features['image']
    resnet = ResnetV2(blocks=[3, 4, 6, 3],
                      block_strides=[2, 2, 2, 1])
    inputs = resnet(inputs=inputs,
                    istraining=istrain,
                    pooling=None)

    rpn = RPN(inputs=inputs, img_shape=_IMG_SHAPE)

    frcnn = FastRCNN(rpn=rpn,
                     num_of_classes=1,
                     is_trainning=istrain)

    # create loss, train_op, predictions
    total_loss = None
    train_op = None
    if istrain or iseval:  # i.e. not in predict mode
        rpnloss = rpn.loss(gtboxes=labels['bbox'])
        frcnnloss = frcnn.loss(gtcls=labels['class'], gtboxes=labels['bbox'])
        total_loss = frcnnloss + rpnloss

        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(total_loss, global_step=globalstep)

        tf.summary.scalar('rpn_loss', rpnloss)
        tf.summary.scalar('frcnn_loss', frcnnloss)
        tf.summary.scalar('sgdlr', sgdlr)

    predictions = None
    if ispredict or iseval:  # i.e. not in train mode
        predictions = frcnn.predict()
        predictions.update({'image': features['image'],
                            'image_id': features['image_id']})

    eval_metric = None
    if iseval:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=total_loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def script_train():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/Test/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    train_input_fn = partial(input_fn, files=train_files, epoch=2, batch=1)
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    lr = 0.01
    momentum = 0.1
    # model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
    #                                config=config,
    #                                params={'sgdlr': lr, 'sgdmomt': momentum})

    # hooks = [tf_debug.LocalCLIDebugHook()]
    # model.train(input_fn=train_input_fn, hooks=hooks)
    for step in range(10):
        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       config=config,
                                       params={'sgdlr': lr,
                                               'sgdmomt': momentum})

        model.train(input_fn=train_input_fn)

        tf.reset_default_graph()
        lr = lr*0.8


if __name__ == '__main__':
    # _script_examine_input()
    script_train()
