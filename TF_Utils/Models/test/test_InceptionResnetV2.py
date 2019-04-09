# -*- coding: utf-8 -*-
"""
Created on 9/28/18
@author: MRChou

Scenario: 
"""

import numpy
import tensorflow as tf


def _test_eval(estimator):
    model = estimator

    def psudo_input_fn():
        data = tf.data.Dataset.from_tensors(
            tf.constant(254.0, shape=(1, 299, 299, 3), dtype=tf.float32))
        data = data.make_one_shot_iterator()
        data = data.get_next()

        wronglabel = tf.constant(1, shape=(1, 19995), dtype=tf.float32)
        rightlabel = tf.sparse_to_dense([1924, 11909, 13801, 14184, 15243,
                                         15245, 16452, 17402, 19338, 19466,
                                         19988, 19989, 19990],
                                        output_shape=[19995],
                                        sparse_values=1.0)
        rightlabel = tf.reshape(rightlabel, [1, 19995])
        return {'input_1': data}, rightlabel

    for i in model.predict(input_fn=psudo_input_fn):
        print(numpy.argwhere(i['output'] > 0.9))

    print(model.evaluate(input_fn=psudo_input_fn))


# used to test function
def _script_examine_input():
    import time
    with tf.Session() as sess:
        image, label = input_fn('/archive/Inclusive/test_TFRs/test_0001.tfrecord')
        image = image['input_1']
        start = time.time()
        for _ in range(1000):
            with tf.device('GPU:0'):
                lbl = sess.run(label)
                print(lbl)
        print('finishd fetching 1000 in %s second' % (time.time()-start))


if __name__ == '__main__':
    pass
