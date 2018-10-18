# -*- coding: utf-8 -*-
"""
Created on 10/13/18
@author: MRChou

Scenario: Trying out a brute force method.
"""

import os
import pathlib
from functools import partial

import tensorflow as tf

from RSNA_Pneumonia.RSNA_DataInput import keras_input_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def _script_examine_input():
    folder = '/archive/RSNA/train_TFRs/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    with tf.Session() as sess:
        feature, label = keras_input_fn(train_files, batch=4, include_neg=False, augment=True)
        for _ in range(10):
            img, lbl = sess.run([feature, label])
            print(lbl)


# work around resetting model_fn for an estimator
def _reset_model_fn(model_fn, estimator):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=estimator.model_dir,
                                  config=estimator.config)


# work around fix for caculating prccurve in evaluation mode
def _model_fn_wrapper(model_fn):
    def _model_fn(features, labels, mode, params):
        imgid = features.pop('image_id')
        spec = model_fn(features, labels, mode, params)
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
    return _model_fn


def keras_inceptresv2(config, sgdlr=0.01, sgdmomt=0.5):
    basemodel = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=[1024, 512, 3],
                                                        pooling='max',
                                                        weights='imagenet')
    featurelayer = basemodel.output
    fc = tf.keras.layers.Dense(2048, activation='relu')(featurelayer)
    dropout = tf.keras.layers.Dropout(rate=0.6)(fc)
    fc = tf.keras.layers.Dense(1024, activation='relu')(dropout)
    dropout = tf.keras.layers.Dropout(rate=0.6)(fc)
    output = tf.keras.layers.Dense(4, activation='relu', name='output')(dropout)

    optimizer = tf.keras.optimizers.SGD(lr=sgdlr, momentum=sgdmomt)

    model = tf.keras.Model(inputs=basemodel.input, outputs=output)
    model.compile(optimizer=optimizer,
                  loss='MSE')
    model = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                  config=config)
    return model


def script_train_keras():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/BBoxer/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(keras_input_fn,
                             files=train_files,
                             epoch=5, batch=4,
                             include_neg=False,
                             augment=True)

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(keras_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            include_neg=False,
                            augment=False)

    lr = 0.01
    momentum = 0.5
    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    session_config=DEVCONFIG)

    for step in range(10):
        model = keras_inceptresv2(config=config, sgdlr=lr, sgdmomt=momentum)
        model_fn = _model_fn_wrapper(model.model_fn)
        model = _reset_model_fn(model_fn, model)

        model.train(input_fn=train_input_fn)
        print(model.evaluate(input_fn=eval_input_fn))
        tf.keras.backend.clear_session()
        lr = lr*0.9


if __name__ == '__main__':
    #_script_examine_input()
    script_train_keras()



