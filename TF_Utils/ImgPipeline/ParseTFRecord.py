# -*- coding: utf-8 -*-
"""
Created on 9/15/18
@author: MRChou

Scenario: utils for configue image pipline from a TFRecrod source.
"""

import tensorflow as tf

from img_feature_proto import OIDPROTO


def _parse_oid_example(example_proto):
    # tfrecords for object detection should match tf.train.Example proto.
    feature_proto = OIDPROTO
    parsed_features = tf.parse_single_example(example_proto, feature_proto)
    return parsed_features


def tfrecord_to_dataset(files, flag='oid'):
    if flag == 'oid':
        map_func = _parse_oid_example
    else:
        msg = 'Unknown flag for reading TFRecord files: {}'.format(flag)
        raise NotImplementedError(msg)
    return tf.data.TFRecordDataset(files).map(map_func=map_func)


if __name__ == '__main__':
    file = "/archive/OpenImg/train_TFRs/train_0002-1000.tfrecord"
    a = tfrecord_to_dataset(file).make_one_shot_iterator()
    a = a.get_next()
    with tf.Session() as sess:
        result = sess.run(a)
        for key in result:
            print('{}: {}'.format(key, result[key]))
