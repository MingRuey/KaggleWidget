# -*- coding: utf-8 -*-
"""
Created on 10/15/18
@author: MRChou

Scenario: Use pretrained hub model to test if RSNA image can be classified.
"""

import os
import pathlib
from functools import partial

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.metrics import recall_at_thresholds as recall
from tensorflow.metrics import precision_at_thresholds as precision

from RSNA_Pneumonia.RSNA_DataInput import keras_input_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True

ir_module_path = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"
pnas_module_path = "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2"
i3_module_path = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'


def ir_input_fn(*args, **kwargs):
    feature, label = keras_input_fn(*args, **kwargs)
    img = tf.image.resize_images(feature['input_1'], size=(299, 299))
    img = img/255
    return {'image': img, 'image_id': feature['image_id']}, label


def pnas_input_fn(*args, **kwargs):
    feature, label = keras_input_fn(*args, **kwargs)
    img = tf.image.resize_images(feature['input_1'], size=(331, 331))
    img = img/255
    return {'image': img, 'image_id': feature['image_id']}, label


def _script_examine_input():
    folder = '/archive/RSNA/train_TFRs/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    with tf.Session() as sess:
        feature, label = ir_input_fn(train_files, batch=4, include_neg=False, augment=True)
        for _ in range(10):
            img, lbl = sess.run([feature, label])
            print(img['image'])


def ir_model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    # load inception resnet v2:
    module = hub.Module(ir_module_path,
                        trainable=istrain,
                        tags={"train"} if istrain else None,
                        name='InceptionResNetV2')

    feature_vector = module(features['image'])

    initer = tf.variance_scaling_initializer()
    with tf.variable_scope('FCs'):
        output = tf.layers.dropout(feature_vector, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024, kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024, kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1, kernel_initializer=initer)

    if istrain or iseval:
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                               logits=output)
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
        predictions = tf.nn.sigmoid(output)
    else:
        predictions = None

    if iseval:
        eval_metric = {'Recalls': recall(labels,
                                         predictions,
                                         (0.1, 0.3, 0.5, 0.7, 0.9)),
                       'Precision': precision(labels,
                                              predictions,
                                              (0.1, 0.3, 0.5, 0.7, 0.9))}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def pnas_model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    # load PNASnet:
    module = hub.Module(pnas_module_path,
                        trainable=istrain,
                        tags={"train"} if istrain else None,
                        name='PNASnet')

    feature_vector = module(features['image'])

    initer = tf.variance_scaling_initializer()
    with tf.variable_scope('FCs'):
        output = tf.layers.dropout(feature_vector, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024, kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024, kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1, kernel_initializer=initer)

    if istrain or iseval:
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                               logits=output)
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
        predictions = tf.nn.sigmoid(output)
    else:
        predictions = None

    if iseval:
        eval_metric = {'Recalls': recall(labels,
                                         predictions,
                                         (0.1, 0.3, 0.5, 0.7, 0.9)),
                       'Precision': precision(labels,
                                              predictions,
                                              (0.1, 0.3, 0.5, 0.7, 0.9))}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def incepv3_model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    # load PNASnet:
    module = hub.Module(i3_module_path,
                        trainable=istrain,
                        tags={"train"} if istrain else None,
                        name='InceptionV3')

    feature_vector = module(features['image'])

    initer = tf.variance_scaling_initializer()
    with tf.variable_scope('FCs'):
        output = tf.layers.dropout(feature_vector, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024, kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1024, kernel_initializer=initer)
        output = tf.layers.dropout(output, rate=0.8, training=istrain)
        output = tf.layers.dense(output, units=1, kernel_initializer=initer)

    if istrain or iseval:
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                               logits=output)
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
        predictions = tf.nn.sigmoid(output)
    else:
        predictions = None

    if iseval:
        eval_metric = {'Recalls': recall(labels,
                                         predictions,
                                         (0.1, 0.3, 0.5, 0.7, 0.9)),
                       'Precision': precision(labels,
                                              predictions,
                                              (0.1, 0.3, 0.5, 0.7, 0.9))}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def script_train(flag):
    folder = '/archive/RSNA/train_TFRs/'

    if flag == 'InceptionResnet':
        model_dir = '/archive/RSNA/models/Classifier_HubIncepRes'
        model_fn = ir_model_fn
        hub_input_fn = ir_input_fn
    elif flag == 'PNASnet':
        model_dir = '/archive/RSNA/models/Classifier_HubPNAS'
        model_fn = pnas_model_fn
        hub_input_fn = pnas_input_fn
    elif flag == 'InceptionV3':
        model_dir = '/archive/RSNA/models/Classifier_HubIncepV3'
        model_fn = incepv3_model_fn
        hub_input_fn = ir_input_fn
    else:
        raise NotImplementedError

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(hub_input_fn,
                             files=train_files,
                             epoch=1, batch=8,
                             include_neg=True,
                             augment=True)

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(hub_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            include_neg=True,
                            augment=False)

    lr = 0.001
    momentum = 0.3
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    for step in range(20):
        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       config=config,
                                       params={'sgdlr': lr,
                                               'sgdmomt': momentum})

        print(model.evaluate(input_fn=eval_input_fn))
        model.train(input_fn=train_input_fn)
        tf.reset_default_graph()
        lr = lr*0.9


if __name__ == '__main__':
    # _script_examine_input()
    script_train('InceptionV3')
