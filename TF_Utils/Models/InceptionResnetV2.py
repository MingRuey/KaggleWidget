# -*- coding: utf-8 -*-
"""
Created on 9/18/18
@author: MRChou

Scenario: convert keras InceptionResnetV2 models into Tensorflow Esitmator.
"""

import pickle
import logging
from pathlib import Path
from functools import partial

import numpy
import tensorflow as tf
from tensorflow.metrics import false_negatives_at_thresholds as tf_fn_at_thres
from tensorflow.metrics import false_positives_at_thresholds as tf_fp_at_thres

from TF_Utils.ImgPipeline.img_feature_proto import CLSPROTO
from TF_Utils.Models.CustomLoss import focal_loss


# monkey patch tf.gradients to openai_gradient
# https://github.com/openai/gradient-checkpointing
# from openai_gradient import memory_saving_gradients
# from tensorflow.python.keras import backend as K
# K.__dict__["gradients"] = memory_saving_gradients.gradients_memory
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

DEVCONFIG = tf.ConfigProto()
DEVCONFIG.gpu_options.allow_growth = True
DEVCONFIG.allow_soft_placement = True


def keras_inceptresv2(config, sgdlr=0.01, sgdmomt=0.5):
    basemodel = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=[None, None, 3],
                                                        pooling='avg',
                                                        weights=None)
    featurelayer = basemodel.output
    dropout = tf.keras.layers.Dropout(rate=0.8)(featurelayer)
    outlayer = tf.keras.layers.Dense(7178,
                                     activation='sigmoid',
                                     name='output')(dropout)

    optimizer = tf.keras.optimizers.SGD(lr=sgdlr, momentum=sgdmomt)
    model = tf.keras.Model(inputs=basemodel.input, outputs=outlayer)
    model.compile(optimizer=optimizer,
                  loss=focal_loss)
    model = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                  config=config)
    return model


def _parse_funtion(example_proto, normalize=True):
    parsed_feature = tf.parse_single_example(example_proto,
                                             features=CLSPROTO)

    imgid = parsed_feature['image/source_id']

    img = tf.image.decode_jpeg(parsed_feature['image/encoded'], channels=3)
    img = tf.image.resize_images(img, [299, 299])
    if normalize:
        img = tf.image.per_image_standardization(img)

    cls = tf.sparse_to_dense(parsed_feature['image/class/index'],
                             output_shape=[7178],
                             sparse_values=1,
                             validate_indices=False)
    return img, cls, imgid


def input_fn(files, batch=1, epoch=1):
    with tf.name_scope('imput_pipleline'):
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_funtion)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        img, cls, imgid = dataset.make_one_shot_iterator().get_next()
    return {'input_1': img}, cls


# work around resetting model_fn for an estimator
def _reset_model_fn(model_fn, estimator):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=estimator.model_dir,
                                  config=estimator.config)


def _input_fn_withid(files, epoch=1, batch=1):
    with tf.name_scope('imput_pipleline'):
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_funtion)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        img, cls, imgid = dataset.make_one_shot_iterator().get_next()

    return {'input_1': img, 'imgid': imgid}, cls


# work around fix for retrieving data ID in predict mode
def _submit_model_fn_wrapper(model_fn, key='imgid'):
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


def script_submit():
    thresholds = [0.7, 0.5, 0.3]
    outfiles = ['/archive/Inclusive/Inclusive00_thres07.csv',
                '/archive/Inclusive/Inclusive00_thres05.csv',
                '/archive/Inclusive/Inclusive00_thres03.csv']

    model_dir = '/archive/Inclusive/models/Inclusive00_InceptResV2'
    folder = '/archive/Inclusive/test_TFRs/'
    prd_files = 'test_*.tfrecord'

    with open('/archive/Inclusive/LABELS_TO_CLSINDEX.pkl', 'rb') as f:
        index_to_label= {value: key for key, value in pickle.load(f).items()}

    def _to_label_string(nparray, thres_value):
        nparray = numpy.argwhere(nparray > thres_value)
        return ' '.join([index_to_label[int(index)] for index in nparray])

    model = keras_inceptresv2(config=tf.estimator.RunConfig(model_dir=model_dir,
                                                            session_config=DEVCONFIG))
    model_fn = _submit_model_fn_wrapper(model.model_fn)
    model = _reset_model_fn(model_fn, model)

    prd_files = [str(path) for path in Path(folder).glob(prd_files)]

    for outfile, thres in zip(outfiles, thresholds):
        with open(outfile, 'w') as fout:
            results = model.predict(input_fn=partial(_input_fn_withid, files=prd_files))

            for result in results:
                imgid, prd = result['imgid'], result['output']
                fout.write(imgid.decode() + ',' + _to_label_string(prd, thres)+'\n')


# work around fix for caculating prccurve in evaluation mode
def _prcurve_model_fn_wrapper(model_fn, thres):
    def model_fn_prcurve(features, labels, mode, params):
        spec = model_fn(features, labels, mode, params)
        output = spec.predictions['output']
        fp = tf_fp_at_thres(labels, output, thres, name='false_positives')
        fn = tf_fn_at_thres(labels, output, thres, name='false_negatives')

        spec.eval_metric_ops.update({'false_positives': fp,
                                     'false_negatives': fn})
        print(spec.eval_metric_ops)

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
    return model_fn_prcurve


def eval_with_prcurve(estimator, files):

    thresholds = [0.1*i for i in range(10)]
    model_fn = _prcurve_model_fn_wrapper(estimator.model_fn, thresholds)
    model = _reset_model_fn(model_fn, estimator)

    metric = model.evaluate(input_fn=partial(input_fn, files=files))
    tf.reset_default_graph()

    return metric


def script_eval():
    folder = '/archive/Inclusive/train_TFRs/'
    model_dir = '/archive/Inclusive/models/model00_ckpt'

    files = 'train_1[6-7][0-9][0-9].tfrecord'
    files = [str(path) for path in Path(folder).glob(files)]

    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    session_config=DEVCONFIG)

    model = keras_inceptresv2(config=config)

    metric = eval_with_prcurve(model, files)
    print(metric)


def script_train_and_eval(evaluate=False):
    folder = '/archive/Inclusive/train_TFRs/'
    model_dir = '/archive/Inclusive/models/Inclusive01_InceptResV2'

    train_files1 = 'train_0[0-9][0-9][0-9].tfrecord'
    train_files2 = 'train_1[0-5][0-9][0-9].tfrecord'
    train_files = [str(path) for path in Path(folder).glob(train_files1)]
    train_files += [str(path) for path in Path(folder).glob(train_files2)]

    eval_folder = '/archive/Inclusive/train_TFRs/'
    eval_files = 'train_1[6-7][0-9][0-9].tfrecord'
    eval_files = [str(path) for path in Path(eval_folder).glob(eval_files)]

    lr = 0.05
    momentum = 0.8
    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    session_config=DEVCONFIG)
    for step in range(10):
        model = keras_inceptresv2(config=config,
                                  sgdlr=lr,
                                  sgdmomt=momentum)

        model.train(input_fn=partial(input_fn,
                                     files=train_files,
                                     batch=8,
                                     epoch=2))
        if evaluate:
            metric = eval_with_prcurve(model, eval_files)
            logging.info(' -- eval %s' % metric)

        tf.reset_default_graph()
        lr = lr*0.8


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    script_eval()
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
    #                     handlers=[logging.FileHandler('InceptResnetV2.log')])
    #
    # script_train_and_eval(evaluate=True)
