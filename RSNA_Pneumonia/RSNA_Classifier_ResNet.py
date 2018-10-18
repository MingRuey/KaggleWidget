# -*- coding: utf-8 -*-
"""
Created on 10/14/18
@author: MRChou

Scenario: 
"""

import os
import pathlib
from functools import partial

import tensorflow as tf
from tensorflow.metrics import recall_at_thresholds as recall
from tensorflow.metrics import precision_at_thresholds as precision

from TF_Utils.Models.Resnet import ResnetV2
from RSNA_Pneumonia.RSNA_DataInput import keras_input_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def resnet50_model_fn(features, labels, mode, params):

    # set global step
    globalstep = tf.train.get_or_create_global_step()

    #  detect mode
    istrain = (mode == tf.estimator.ModeKeys.TRAIN)
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    resnet = ResnetV2(blocks=[3, 4, 6, 3],
                      block_strides=[2, 2, 2, 1])

    inputs = features['input_1']
    inputs = resnet(inputs=inputs,
                    istraining=istrain)

    output = tf.layers.dense(inputs, 1,
                             trainable=True,
                             name='BaseModelFC')

    # create loss, train_op, predictions
    if istrain or iseval:  # i.e. not in predict mode
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                               logits=output)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

        # make sure that batch norm works properly
        # https://stackoverflow.com/questions/43234667/
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


def script_train_resnet():
    model_dir = '/archive/RSNA/models/Classifier_Res50V2/'
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)
    resnet = tf.estimator.Estimator(model_fn=resnet50_model_fn,
                                    model_dir=model_dir, config=config)

    folder = '/archive/RSNA/train_TFRs/'
    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(keras_input_fn, files=train_files, epoch=50, batch=8)
    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(keras_input_fn, files=eval_files, epoch=1, batch=1)

    resnet.train(input_fn=train_input_fn)
    print(resnet.evaluate(input_fn=eval_input_fn))


if __name__ == '__main__':
    # script_train_resnet()
    pass
