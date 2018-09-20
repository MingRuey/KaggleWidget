# -*- coding: utf-8 -*-
"""
Created on 9/18/18
@author: MRChou

Scenario: CNN with tf.estimator API.
"""

from pathlib import Path
from functools import partial

import tensorflow as tf

from openai_gradient import memory_saving_gradients
from TF_Utils.ImgPipeline.img_feature_proto import CLSPROTO

# monkey patch tf.gradients to openai_gradient
# https://github.com/openai/gradient-checkpointing
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory


def keras_inceptresv2(model_dir):
    basemodel = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=[None, None, 3],
                                                        pooling='avg',
                                                        weights=None)
    featurelayer = basemodel.output
    dropout = tf.keras.layers.Dropout(rate=0.8)(featurelayer)
    outlayer = tf.keras.layers.Dense(19995,
                                     activation='sigmoid',
                                     name='output')(dropout)

    model = tf.keras.Model(inputs=basemodel.input, outputs=outlayer)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy')

    model = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                  model_dir=model_dir)

    return model


def _parse_funtion(example_proto):
    parsed_feature = tf.parse_single_example(example_proto,
                                             features=CLSPROTO)
    img = tf.image.decode_jpeg(parsed_feature['image/encoded'],
                               channels=3)
    cls = tf.sparse_to_dense(parsed_feature['image/class/index'],
                             output_shape=[19995],
                             sparse_values=1,
                             validate_indices=False)
    return img, cls


def input_fn(file_path, batch=1, epoch=1):
    with tf.name_scope('imput_pipleline'):
        dataset = tf.data.TFRecordDataset(file_path, buffer_size=0)
        dataset = dataset.map(_parse_funtion)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch)
        img, cls = dataset.make_one_shot_iterator().get_next()
    return {'input_1': img}, cls


# TODO: a feature column definer
def feature_column_fn():
    some_feature_column = None
    return some_feature_column


# TODO: Instantiate an estimator
def get_an_estimator(feature_column=None):
    some_estimator = None
    return some_estimator


# TODO: Some trainning function
def train(estimator=None, config=None):
    pass


# used to test function
def test_input():
    with tf.Session() as sess:
        image, label = input_fn('/archive/Inclusive/train_TFRs/train_0001.tfrecord')
        print(sess.run(image['input_1']).shape)
        print(sess.run(label).shape)


if __name__ == '__main__':
    folder = '/archive/Inclusive/train_TFRs/'
    model_dir = '/archive/Inclusive/models/'

    train_files = 'train_01*.tfrecord'
    train_files = [str(path) for path in Path(folder).glob(train_files)]

    model = keras_inceptresv2(model_dir=model_dir)
    model.train(input_fn=partial(input_fn, file_path=train_files))

    # ---

    eval_file = 'train_020*.tfrecord'
    eval_file = [str(path) for path in Path(folder).glob(eval_file)]

    result = model.evaluate(input_fn=partial(input_fn, file_path=eval_file))
                            #checkpoint_path=model_dir + 'beforetrain/keras_model.ckpt')

    print(result)
