# -*- coding: utf-8 -*-
"""
Created on 10/16/18
@author: MRChou

Scenario: Trying out a brute force method with tf hub pretrained model.
          Seperate box position / (width, height) prediction into two FCs.
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
    ymin, xmin, ymax, xmax = tf.split(label, 4, axis=1)
    return {'image': img, 'image_id': feature['image_id']}, \
           {'Position': tf.concat([ymin, xmin], axis=-1),
            'Shape': tf.concat([ymax-ymin, xmax-xmin], axis=-1)}


def _script_examine_input():
    folder = '/archive/RSNA/train_TFRs/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    with tf.Session() as sess:
        feature, label = hub_input_fn(train_files, batch=4, augment='horizontal', include_neg=False)
        for _ in range(15):
            img, cls = sess.run([feature, label])
            print(cls['Position'], cls['Position'].shape)
            print(cls['Shape'], cls['Shape'].shape)


def model_fn(features, labels, mode, params):
    globalstep = tf.train.get_or_create_global_step()
    istrain = mode == tf.estimator.ModeKeys.TRAIN
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    # load inception resnet v2:
    module = hub.Module(module_path,
                        trainable=True,
                        name='InceptionResNetV2')

    with tf.variable_scope('InceptResV2'):
        feature_vector = module(features['image'])

    initer = tf.variance_scaling_initializer()
    with tf.variable_scope('Position_FCs'):
        pos = tf.layers.dropout(feature_vector, rate=0.8, training=istrain)
        pos = tf.layers.dense(pos,
                              units=1024,
                              activation=tf.nn.relu,
                              kernel_initializer=initer)
        pos = tf.layers.dropout(pos, rate=0.8, training=istrain)
        pos = tf.layers.dense(pos,
                              units=1024,
                              activation=tf.nn.relu,
                              kernel_initializer=initer)
        pos = tf.layers.dropout(pos, rate=0.8, training=istrain)
        pos = tf.layers.dense(pos,
                              units=2,
                              activation=tf.sigmoid,
                              kernel_initializer=initer)

    with tf.variable_scope('Shape_FCs'):
        shape = tf.layers.dropout(feature_vector, rate=0.8, training=istrain)
        shape = tf.layers.dense(shape,
                                units=1024,
                                activation=tf.nn.relu,
                                kernel_initializer=initer)
        shape = tf.layers.dropout(shape, rate=0.8, training=istrain)
        shape = tf.layers.dense(shape, units=1024,
                                activation=tf.nn.relu,
                                kernel_initializer=initer)
        shape = tf.layers.dropout(shape, rate=0.8, training=istrain)
        shape = tf.layers.dense(shape,
                                units=2,
                                activation=tf.sigmoid,
                                kernel_initializer=initer)

    if istrain or iseval:
        pos_loss = 5*tf.losses.mean_squared_error(labels=labels['Position'],
                                                  predictions=pos)
        shape_loss = tf.losses.mean_squared_error(labels=labels['Shape'],
                                                  predictions=shape)
        total_loss = pos_loss + shape_loss
        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(total_loss, global_step=globalstep)

        tf.summary.scalar('pos_loss', pos_loss)
        tf.summary.scalar('shape_loss', shape_loss)
    else:
        total_loss = None
        train_op = None

    output = tf.concat([pos, pos+shape], axis=-1)
    if ispredict or iseval:  # i.e. not in train mode
        predictions = {'output': output,
                       'image_id': features['image_id']}

    else:
        predictions = None

    if iseval:
        gt_boxes = tf.concat([labels['Position'],
                              labels['Shape']+labels['Position']], axis=-1)
        eval_metric = {'mean_iou': tf.metrics.mean(
            tf.reduce_max(walkerlala_iou(output, gt_boxes), axis=1)
        )}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=total_loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def script_train_hub():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/Locator/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(hub_input_fn,
                             files=train_files,
                             epoch=1, batch=1,
                             include_neg=False,
                             augment='horizontal')

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(hub_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            include_neg=False,
                            augment=False)

    lr = 0.05
    momentum = 0.5
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    for step in range(50):
        model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                       config=config,
                                       params={'sgdlr': lr,
                                               'sgdmomt': momentum})

        model.train(input_fn=train_input_fn)
        print(model.evaluate(input_fn=eval_input_fn))
        tf.reset_default_graph()
        lr = lr*0.95


if __name__ == '__main__':
    # _script_examine_input()
    script_train_hub()