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
from tensorflow.metrics import recall_at_thresholds as recall
from tensorflow.metrics import precision_at_thresholds as precision

from RSNA_Pneumonia.RSNA_DataInput import keras_input_fn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def _script_examine_input():
    folder = '/archive/RSNA/train_TFRs/'

    train_files = 'train_00[0-9][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]

    with tf.Session() as sess:
        feature, label = keras_input_fn(train_files, batch=8, include_neg=False)
        for _ in range(15):
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

        spec.predictions.update({'image_id': imgid})

        if mode == tf.estimator.ModeKeys.EVAL:
            output = spec.predictions['output']
            eval_metric = {'Recalls': recall(labels,
                                             output,
                                             (0.1, 0.3, 0.5, 0.7, 0.9)),
                           'Precision': precision(labels,
                                                  output,
                                                  (0.1, 0.3, 0.5, 0.7, 0.9))}

            spec.eval_metric_ops.update(eval_metric)

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
                                                        input_shape=[1024, 512, 3],
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


def script_train_keras():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/Classifier/KerasIncepRes/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    train_files = [str(path) for path in pathlib.Path(folder).glob(train_files)]
    train_input_fn = partial(keras_input_fn,
                             files=train_files,
                             epoch=5, batch=4,
                             include_neg=True,
                             augment=True)

    eval_files = 'train_002[0-6].tfrecord'
    eval_files = [str(path) for path in pathlib.Path(folder).glob(eval_files)]
    eval_input_fn = partial(keras_input_fn,
                            files=eval_files,
                            epoch=1, batch=1,
                            include_neg=True,
                            augment=False)

    lr = 0.05
    momentum = 0.5
    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    session_config=DEVCONFIG)

    for step in range(10):
        model = keras_inceptresv2(config=config, sgdlr=lr, sgdmomt=momentum)
        model_fn = _acc_model_fn_wrapper(model.model_fn)
        model = _reset_model_fn(model_fn, model)

        print(model.evaluate(input_fn=eval_input_fn))
        model.train(input_fn=train_input_fn)
        tf.keras.backend.clear_session()
        lr = lr*0.9


if __name__ == '__main__':
    _script_examine_input()
    # script_train_keras()
