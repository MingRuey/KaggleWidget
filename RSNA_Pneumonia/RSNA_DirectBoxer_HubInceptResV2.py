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
import tensorflow_hub as hub

from RSNA_Pneumonia.RSNA_DataInput import keras_input_fn
from TF_Utils.Models.FasterRCNN.regression_anchors import walkerlala_iou

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True

_img_shape = (299, 299)
module_path = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"


def hub_input_fn(*args, **kwargs):
    feature, label = keras_input_fn(*args, **kwargs)
    img = tf.image.resize_images(feature['input_1'], size=_img_shape)
    img = img/255
    return {'image': img, 'image_id': feature['image_id'], 'labels': label}, label


def model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    # load inception resnet v2:
    module = hub.Module(module_path, trainable=True)

    with tf.variable_scope('InceptResV2'):
        feature_vector = module(features['image'])

    initer = tf.variance_scaling_initializer()
    with tf.variable_scope('FCs'):
        output = tf.layers.dropout(feature_vector, rate=0.8, training=istrain)
        output = tf.layers.dense(output,
                                 units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output,
                                 units=4,
                                 activation=tf.sigmoid,
                                 kernel_initializer=initer)

    if istrain or iseval:
        loss = tf.losses.mean_squared_error(labels=labels,
                                            predictions=output)
        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, global_step=globalstep)
    else:
        loss = None
        train_op = None

    if ispredict or iseval:  # i.e. not in train mode
        predictions = {'output': output,
                       'image_id': features['image_id']}
    else:
        predictions = None

    if iseval:
        eval_metric = {'mean_iou': tf.metrics.mean(
            tf.reduce_max(walkerlala_iou(output, labels), axis=1)
        )}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def script_train_hub():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/BBoxer_resize/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(hub_input_fn,
                             files=train_files,
                             epoch=3, batch=8,
                             include_neg=False,
                             augment='horizontal')

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(hub_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            include_neg=False,
                            augment=False)

    lr = 0.04
    momentum = 0.5
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    for step in range(10):
        model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                       config=config,
                                       params={'sgdlr': lr,
                                               'sgdmomt': momentum})

        model.train(input_fn=train_input_fn)
        print(model.evaluate(input_fn=eval_input_fn))
        tf.reset_default_graph()
        lr = lr*0.95


if __name__ == '__main__':
    script_train_hub()
