# -*- coding: utf-8 -*-
"""
Created on 10/11/18
@author: MRChou

Scenario: scripts for testing Resnet.py.
"""

import pathlib
import xml.etree.ElementTree as ET
from functools import partial

import cv2
import numpy
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from TF_Utils.Models.Resnet import ResnetV2

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True

_IMG_SHAPE = (600, 600)
CLS_TO_INT = {'banana': 0, 'apple': 1, 'orange': 2}
INT_TO_CLS = {v: k for v, k in CLS_TO_INT.items()}


def _parse_data(imgfile, xmlfile):
    imgid = pathlib.Path(imgfile.decode()).stem
    img = cv2.imread(imgfile.decode())[..., ::-1]
    cls = [CLS_TO_INT[ET.parse(xmlfile).getroot().find('object').find('name').text]]
    img = cv2.resize(img, _IMG_SHAPE)
    img = img.astype(numpy.float32)
    return imgid, img, cls


def data_input(path, batch=1, epoch=1):

    def files_gener():
        img_files = (img for img in pathlib.Path(path).iterdir() if str(img).endswith('.jpg'))
        for img_file in img_files:
            yield path+img_file.stem+'.jpg', path+img_file.stem+'.xml'

    dataset = tf.data.Dataset.from_generator(files_gener, (tf.string, tf.string))
    dataset = dataset.map(lambda image, xml: tf.py_func(_parse_data,
                                                        [image, xml],
                                                        [tf.string, tf.float32, tf.int64]))
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch)
    imgid, img, cls = dataset.make_one_shot_iterator().get_next()
    img.set_shape((None, _IMG_SHAPE[0], _IMG_SHAPE[1], 3))

    return {'image_id': imgid, 'image': img}, cls[0]


def mnist_input(batch=1, epoch=1, mode='train'):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    if mode == 'train':
        mnist = tf.data.Dataset.from_tensor_slices((mnist.train.images,
                                                    mnist.train.labels))
    elif mode == 'validation':
        mnist = tf.data.Dataset.from_tensor_slices((mnist.validation.images,
                                                    mnist.validation.labels))
    else:
        mnist = tf.data.Dataset.from_tensor_slices((mnist.test.images,
                                                    mnist.test.labels))
    mnist = mnist.repeat(epoch)
    mnist = mnist.batch(batch)
    image, label = mnist.make_one_shot_iterator().get_next()
    image = tf.reshape(image, shape=[-1, 28, 28, 1])
    image = tf.image.resize_images(image, size=[224, 224])
    label = tf.cast(label, tf.float32)
    return {'image': image, 'label': label}, label


def _script_examine_input():
    image, label = mnist_input()
    with tf.Session() as sess:
        for _ in range(5):
            img, lbl = sess.run([image, label])
            print(lbl.dtype)


def keras_input(batch=1, epoch=1, mode='train'):
    img_dict, cls = mnist_input(batch=batch, epoch=epoch, mode=mode)
    return {'input_1': img_dict['image'], 'label': img_dict['label']}, cls


# work around resetting model_fn for an estimator
def _reset_model_fn(model_fn, estimator):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=estimator.model_dir,
                                  config=estimator.config)


# work around fix for retrieving data ID in predict mode
def _pred_model_fn_wrapper(model_fn, key='label'):
    def model_fn_gotkey(features, labels, mode, params):
        key_tensor = features.pop(key)
        spec = model_fn(features, labels, mode, params)
        spec.predictions.update({key: key_tensor})
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
    return model_fn_gotkey


def keras_resnet(config):
    basemodel = tf.keras.applications.ResNet50(include_top=False,
                                               weights=None,
                                               input_shape=(224, 224, 1),
                                               pooling='avg')

    featurelayer = basemodel.output
    outlayer = tf.keras.layers.Dense(10,
                                     activation='softmax',
                                     name='output')(featurelayer)

    model = tf.keras.Model(inputs=basemodel.input, outputs=outlayer)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
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

    resnet = ResnetV2(blocks=[3, 4, 6, 3],
                      block_strides=[2, 2, 2, 1])

    inputs = features['image']
    inputs = resnet(inputs=inputs,
                    istraining=istrain)

    inputs = tf.layers.dense(inputs, 10,
                             trainable=True,
                             name='FinalFC')

    # create loss, train_op, predictions
    if istrain or iseval:  # i.e. not in predict mode
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                               logits=inputs)
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
        predictions = inputs
    else:
        predictions = None

    if iseval:
        eval_metric = {'accuracy': tf.metrics.accuracy(tf.argmax(labels, axis=-1),
                                                       tf.argmax(predictions, axis=-1))}
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def _script_keras_test():
    model_dir = '/archive/RSNA/models/Test_Keras/'
    # train_path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/train/'
    # test_path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/test/'
    #
    # pred_input_fn = partial(data_input, path=test_path, epoch=1)

    resnet = keras_resnet(config=tf.estimator.RunConfig(model_dir=model_dir,
                                                        session_config=DEVCONFIG))

    model_fn = _pred_model_fn_wrapper(resnet.model_fn)
    resnet = _reset_model_fn(model_fn, resnet)

    train_input = partial(keras_input, epoch=5, batch=100, mode='train')
    test_input = partial(keras_input, mode='test')

    # resnet.train(input_fn=train_input)
    # print(resnet.evaluate(input_fn=test_input))
    for result in resnet.predict(input_fn=test_input, yield_single_examples=False):
        print(result['label'])
        print(result['output'])


def _script_test_basemodel():
    model_dir = '/archive/RSNA/models/Test_raw/'
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)
    # train_path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/train/'
    # test_path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/test/'
    #
    # train_input_fn = partial(data_input, path=train_path, epoch=100)
    # pred_input_fn = partial(data_input, path=test_path, epoch=1)

    resnet = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir=model_dir,
                                    config=config)

    train_input = partial(mnist_input, epoch=3, batch=100, mode='train')
    test_input = partial(mnist_input, mode='test')

    resnet.train(input_fn=train_input)
    print(resnet.evaluate(input_fn=test_input))


if __name__ == '__main__':
    # _script_examine_input()
    # _script_keras_test()
    _script_test_basemodel()


