# -*- coding: utf-8 -*-
"""
Created on 10/9/18
@author: MRChou

Scenario: test run rpn on fruit data.
"""

import pathlib
import xml.etree.ElementTree as ET
from functools import partial

import numpy
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from TF_Utils.Models.Resnet import ResnetV2
from TF_Utils.Models.FasterRCNN.RPN import RPN
from TF_Utils.Models.FasterRCNN.FRCNN import FastRCNN

_IMG_SHAPE = (600, 600)

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True

CLS_TO_INT = {'banana': 1, 'apple': 2, 'orange': 3}
INT_TO_CLS = {v: k for k, v in CLS_TO_INT.items()}


def _tune_bndbox(bndbox, original_img_shape):
    ymin = float(bndbox.find('ymin').text)*_IMG_SHAPE[0]/original_img_shape[0]
    ymax = float(bndbox.find('ymax').text)*_IMG_SHAPE[0]/original_img_shape[0]
    xmin = float(bndbox.find('xmin').text)*_IMG_SHAPE[1]/original_img_shape[1]
    xmax = float(bndbox.find('xmax').text)*_IMG_SHAPE[1]/original_img_shape[1]
    return [numpy.float32(ymin), numpy.float32(xmin),
            numpy.float32(ymax), numpy.float32(xmax)]


def _parse_data(imgfile, xmlfile):
    imgid = pathlib.Path(imgfile.decode()).stem
    img = cv2.imread(imgfile.decode())[..., ::-1]
    labels = []
    cls = []
    for obj in ET.parse(xmlfile).getroot().findall('object'):
        bndbox = obj.find('bndbox')
        bndbox = _tune_bndbox(bndbox, img.shape)
        cls.append(CLS_TO_INT[obj.find('name').text])
        labels.append(bndbox)
    img = cv2.resize(img, _IMG_SHAPE)
    img = img.astype(numpy.float32)
    return imgid, img, labels, cls


def data_input(path, batch=1, epoch=1):

    def files_gener():
        img_files = (img for img in pathlib.Path(path).iterdir() if str(img).endswith('.jpg'))
        for img_file in img_files:
            yield path+img_file.stem+'.jpg', path+img_file.stem+'.xml'

    dataset = tf.data.Dataset.from_generator(files_gener, (tf.string, tf.string))
    dataset = dataset.map(lambda image, xml: tf.py_func(_parse_data,
                                                        [image, xml],
                                                        [tf.string, tf.float32, tf.float32, tf.int64]))
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch)
    imgid, img, bbox, cls = dataset.make_one_shot_iterator().get_next()
    img.set_shape((None, _IMG_SHAPE[0], _IMG_SHAPE[1], 3))

    return {'image_id': imgid, 'image': img, 'bbox': bbox[0]}, {'bbox': bbox[0], 'class': cls[0]}


def _script_examine_input():
    import time
    path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/train/'

    with tf.Session() as sess:
        image, labels = data_input(path)
        start = time.time()
        for _ in range(1000):
            img, lbl = sess.run([image, labels])
            print(img['image'].dtype)
        print('finishd fetching in %s second' % (time.time()-start))


def model_fn(features, labels, mode, params):
    # set global step
    globalstep = tf.train.get_or_create_global_step()

    # detect mode
    istrain = (mode == tf.estimator.ModeKeys.TRAIN)
    iseval = (mode == tf.estimator.ModeKeys.EVAL)
    ispredict = (mode == tf.estimator.ModeKeys.PREDICT)

    resnet = ResnetV2(blocks=[3, 4, 6, 3],
                      block_strides=[2, 2, 2, 1])
    inputs = features['image']
    inputs = resnet(inputs=inputs,
                    istraining=istrain,
                    pooling=None)

    rpn = RPN(inputs=inputs, img_shape=_IMG_SHAPE)

    frcnn = FastRCNN(rpn=rpn,
                     num_of_classes=3,
                     is_trainning=istrain)

    # create loss, train_op, predictions
    if istrain or iseval:  # i.e. not in predict mode
        rpnloss = rpn.loss(gtboxes=labels['bbox'])
        frcnnloss = frcnn.loss(gtcls=labels['class'], gtboxes=labels['bbox'])
        total_loss = rpnloss + frcnnloss
        sgdlr = 0.01 if 'sgdlr' not in params else params['sgdlr']
        sgdmomt = 0.5 if 'sgdmomt' not in params else params['sgdmomt']
        optimizer = tf.train.MomentumOptimizer(learning_rate=sgdlr,
                                               momentum=sgdmomt)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(total_loss, global_step=globalstep)
    else:
        total_loss = None
        train_op = None

    if ispredict or iseval:  # i.e. not in train mode
        predictions = frcnn.predict()
        predictions.update({'image': features['image'],
                            'image_id': features['image_id']})
    else:
        predictions = None

    if iseval:
        eval_metric = None
    else:
        eval_metric = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=total_loss,
                                      predictions=predictions,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric)


def script_test_predict():
    train_path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/train/'
    test_path = '/rawdata/_ToyDataSet/FruitImg_ObjDetection/test/'
    model_dir = '/archive/RSNA/models/Test/'

    train_input_fn = partial(data_input, path=train_path, epoch=100)
    pred_input_fn = partial(data_input, path=test_path, epoch=1)
    config = tf.estimator.RunConfig(session_config=DEVCONFIG)

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=model_dir,
                                   config=config
                                   )

    # model.train(input_fn=train_input_fn)

    # numpy.set_printoptions(threshold=numpy.nan)
    # for result in model.predict(input_fn=train_input_fn, yield_single_examples=False):
    #     print(result.shape)
    #     print(numpy.count_nonzero(result == 0))
    #     print(numpy.count_nonzero(result == 1))
    #     print(numpy.count_nonzero(result == -1))
    for result in model.predict(input_fn=pred_input_fn):
        imgid = result['image_id'].decode()
        img = result['image']

        fig, ax = plt.subplots(1)
        ax.imshow(img.astype(numpy.uint8))
        ax.text(0, 0, imgid)

        for bbox, cls, prob in zip(result['bbox'],
                                   result['class'],
                                   result['probability']):

            ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1, edgecolor='r',
                                     facecolor='none')

            ax.add_patch(rect)
            ax.text(xmin, ymin, INT_TO_CLS[cls] + ':' + str(prob))

        plt.show()


if __name__ == '__main__':
    _script_examine_input()
    # script_test_predict()

