# -*- coding: utf-8 -*-
"""
Created on 10/12/18
@author: MRChou

Scenario: Use pretrained keras model to test if RSNA image can be classified.
"""

import os
import pathlib
from functools import partial

import tensorflow as tf

from TF_Utils.ImgPipeline.img_feature_proto import OIDPROTO

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True

_IMG_SHAPE = (299, 299)


def _parse_func(example):
    parsed_feature = tf.parse_single_example(example, features=OIDPROTO)

    imgid = parsed_feature['image/source_id']

    # decode img and change dtype to float32
    img = parsed_feature['image/encoded']
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, size=_IMG_SHAPE)
    img = tf.cast(img, dtype=tf.float32)

    cls = parsed_feature['image/object/class/index'][0]
    cls = tf.reshape(cls, [1])

    return imgid, img, cls


def keras_input_fn(files, epoch=1, batch=1):
    with tf.name_scope('imput_pipleline'):
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_func)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        dataset = dataset.make_one_shot_iterator()
        imgid, img, cls = dataset.get_next()

    return {'input_1': img, 'image_id': imgid}, cls


def _script_examine_input():
    folder = '/archive/RSNA/train_TFRs/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    with tf.Session() as sess:
        feature, label = keras_input_fn(train_files, batch=8)
        for _ in range(10):
            img, cls = sess.run([feature, label])
            print(cls, cls.shape)


# work around resetting model_fn for an estimator
def _reset_model_fn(model_fn, estimator):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=estimator.model_dir,
                                  config=estimator.config)


# work around fix for caculating prccurve in evaluation mode
def _acc_model_fn_wrapper(model_fn):
    def model_fn_acc(features, labels, mode, params):
        imgid = features.pop('image_id')
        spec = model_fn(features, labels, mode, params)

        output = spec.predictions['output']
        _ones = tf.ones_like(output)
        _zeros = tf.zeros_like(output)
        pred_at_03 = tf.where(output > 0.3, _ones, _zeros)
        pred_at_05 = tf.where(output > 0.5, _ones, _zeros)
        pred_at_07 = tf.where(output > 0.7, _ones, _zeros)
        eval_metric = {'acc @ 0.3': tf.metrics.accuracy(labels, pred_at_03),
                       'acc @ 0.5': tf.metrics.accuracy(labels, pred_at_05),
                       'acc @ 0.7': tf.metrics.accuracy(labels, pred_at_07)}

        spec.eval_metric_ops.update(eval_metric)
        spec.predictions.update({'image_id': imgid})

        return tf.estimator.EstimatorSpec(mode=spec.mode,
                                          loss=spec.loss,
                                          predictions=spec.predictions,
                                          train_op=spec.train_op,
                                          eval_metric_ops=spec.eval_metric_ops,
                                          export_outputs=spec.export_outputs,
                                          training_chief_hooks=spec.training_chief_hooks,
                                          training_hooks=spec.training_hooks,
                                          scaffold=spec. scaffold,
                                          evaluation_hooks=spec.evaluation_hooks,
                                          prediction_hooks=spec.prediction_hooks)
    return model_fn_acc


def keras_inceptresv2(config, sgdlr=0.01, sgdmomt=0.5):
    basemodel = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=[299, 299, 3],
                                                        pooling='avg',
                                                        weights='imagenet')
    featurelayer = basemodel.output
    dropout = tf.keras.layers.Dropout(rate=0.8)(featurelayer)
    outlayer = tf.keras.layers.Dense(1,
                                     activation='sigmoid',
                                     name='output')(dropout)

    optimizer = tf.keras.optimizers.SGD(lr=sgdlr, momentum=sgdmomt)
    model = tf.keras.Model(inputs=basemodel.input, outputs=outlayer)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy')
    model = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                  config=config)
    return model


def script_train():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/Classify/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(keras_input_fn, files=train_files, epoch=5, batch=8)

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(keras_input_fn, files=eval_files, epoch=1, batch=8)

    lr = 0.01
    momentum = 0.3
    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    session_config=DEVCONFIG)

    for step in range(10):
        model = keras_inceptresv2(config=config, sgdlr=lr, sgdmomt=momentum)
        model_fn = _acc_model_fn_wrapper(model.model_fn)
        model = _reset_model_fn(model_fn, model)

        model.train(input_fn=train_input_fn)
        tf.keras.backend.clear_session()
        lr = lr*0.9


if __name__ == '__main__':
    # _script_examine_input()
    script_train()
