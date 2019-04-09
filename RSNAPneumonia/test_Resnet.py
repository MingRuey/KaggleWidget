# -*- coding: utf-8 -*-
"""
Created on 9/24/18
@author: MRChou

Scenario: scripts for testing Resnet.py.
"""

import os
from pathlib import Path
from functools import partial

import tensorflow as tf

from TF_Utils.Models.Resnet import ResnetV2
from TF_Utils.ImgPipeline.img_feature_proto import OIDPROTO

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def _parse_funtion(example_proto, normalize=True):
    parsed_feature = tf.parse_single_example(example_proto,
                                             features=OIDPROTO)

    imgid = parsed_feature['image/source_id']

    img = tf.image.decode_jpeg(parsed_feature['image/encoded'], channels=1)
    img = tf.image.resize_images(img, [256, 256])
    if normalize:
        img = tf.image.per_image_standardization(img)

    cls = parsed_feature['image/object/class/index'][0]
    return img, cls, imgid


def _cls_filter(cls_value, *features):
    img, cls, imgid = features
    return tf.equal(cls, cls_value)


def input_fn(files, batch=1, epoch=1, filter_value=None):
    with tf.name_scope('imput_pipleline'):
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_funtion)
        if filter_value is not None:
            dataset = dataset.filter(partial(_cls_filter, filter_value))
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        img, cls, imgid = dataset.make_one_shot_iterator().get_next()
    return {'image': img}, cls


def _script_examine_input():
    import time
    with tf.Session() as sess:
        folder = '/archive/RSNA/train_TFRs/'
        files = 'train_002[0-9].tfrecord'
        files = [str(path) for path in Path(folder).glob(files)]
        image, label = input_fn(files=files, filter_value=1)
        image = image['image']
        start = time.time()
        for _ in range(1000):
            img, lbl = sess.run([image, label])
            print(lbl)
        print('finishd fetching in %s second' % (time.time()-start))


def keras_input(files, batch=1, epoch=1, filter_value=None):
    img_dict, cls = input_fn(files,
                             batch=batch, epoch=epoch,
                             filter_value=filter_value)
    cls = tf.cast(cls, tf.float32)
    cls = tf.expand_dims(cls, -1)
    return {'input_1': img_dict['image']}, cls


def keras_resnet(config):
    basemodel = tf.keras.applications.ResNet50(include_top=False,
                                               weights=None,
                                               input_shape=(256, 256, 1),
                                               pooling='avg')

    featurelayer = basemodel.output
    outlayer = tf.keras.layers.Dense(1,
                                     activation='sigmoid',
                                     name='output')(featurelayer)

    optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.5)
    model = tf.keras.Model(inputs=basemodel.input, outputs=outlayer)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy')
    model = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                  config=config)
    return model


def model_fn(features, labels, mode, params):
    # set global step
    globalstep = tf.train.get_or_create_global_step()

    #  detect mode
    istrain = (mode == tf.estimator.ModeKeys.TRAIN)
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    resnet = ResnetV2(blocks=[3, 4, 23, 3],
                      block_strides=[2, 2, 2, 1])

    inputs = features['image']
    inputs = resnet(inputs=inputs,
                    istraining=istrain)

    inputs = tf.layers.dense(inputs, 1,
                             activation=tf.nn.relu,
                             trainable=True,
                             name='FinalFC')

    # create loss, train_op, predictions
    if istrain or iseval:  # i.e. not in predict mode
        labels = tf.cast(labels, tf.float32)  # for computing loss and acc
        loss = tf.losses.sigmoid_cross_entropy(tf.expand_dims(labels, -1),
                                               logits=inputs,
                                               scope='Loss')
        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)
        train_op = optimizer.minimize(loss, global_step=globalstep)
    else:
        loss = None
        train_op = None

    if ispredict or iseval:  # i.e. not in train mode
        predictions = tf.sigmoid(inputs, name='Output')
    else:
        predictions = None

    if iseval:
        _ones = tf.ones_like(predictions)
        _zeros = tf.zeros_like(predictions)
        pred_at_04 = tf.where(predictions > 0.4, _ones, _zeros)
        pred_at_06 = tf.where(predictions > 0.6, _ones, _zeros)
        pred_at_08 = tf.where(predictions > 0.8, _ones, _zeros)
        eval_metric = {'acc @ 0.4': tf.metrics.accuracy(labels, pred_at_04),
                       'acc @ 0.6': tf.metrics.accuracy(labels, pred_at_06),
                       'acc @ 0.8': tf.metrics.accuracy(labels, pred_at_08)}

    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def _script_test_model_fn():
    tf.logging.set_verbosity(tf.logging.INFO)

    files = '/archive/RSNA/train_TFRs/train_0001.tfrecord'
    model_dir = '/archive/RSNA/models/test/'

    resnet = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir=model_dir)

    inputs = partial(input_fn, files=files)
    predict = [pred for pred in resnet.predict(input_fn=inputs)]
    evaluate = resnet.evaluate(input_fn=inputs)
    print(predict)
    print(evaluate)


def _script_keras_test():
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/test/'

    train_files = 'train_00[0-1][0-9].tfrecord'
    eval_files = 'train_002[0-9].tfrecord'
    train_files = [str(path) for path in Path(folder).glob(train_files)]
    eval_files = [str(path) for path in Path(folder).glob(eval_files)]

    train_inputs = partial(keras_input, files=train_files, batch=8, epoch=100,
                           filter_value=1)
    eval_inputs = partial(keras_input, files=eval_files)

    resnet = keras_resnet(config=tf.estimator.RunConfig(model_dir=model_dir,
                                                        session_config=DEVCONFIG))

    # resnet.train(input_fn=train_inputs)
    predict = [pred for pred in resnet.predict(input_fn=eval_inputs)]
    print(predict)

    evaluate = resnet.evaluate(input_fn=eval_inputs)
    print(evaluate)


def script_train_and_eval_basemodel(evaluate=True):
    folder = '/archive/RSNA/train_TFRs/'
    model_dir = '/archive/RSNA/models/BaseRes101V2'

    train_files = 'train_00[0-1][0-9].tfrecord'
    eval_files = 'train_002[0-9].tfrecord'
    train_files = [str(path) for path in Path(folder).glob(train_files)]
    eval_files = [str(path) for path in Path(folder).glob(eval_files)]

    train_inputs = partial(input_fn, files=train_files, batch=8, epoch=25)
    eval_inputs = partial(input_fn, files=eval_files)

    lr = 0.05
    momentum = 0.5
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    for step in range(10):
        resnet = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_dir,
                                        params={'sgdlr': lr,
                                                'sgdmomt': momentum},
                                        config=config)

        resnet.train(input_fn=train_inputs)

        if evaluate:
            resnet.evaluate(input_fn=eval_inputs)

        tf.reset_default_graph()
        lr = lr*0.8


if __name__ == '__main__':
    # _script_examine_input()
    # _script_test_model_fn()
    # _script_keras_test()
    script_train_and_eval_basemodel()


