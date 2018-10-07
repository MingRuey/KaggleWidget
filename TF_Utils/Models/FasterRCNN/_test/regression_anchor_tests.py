# -*- coding: utf-8 -*-
"""
Created on 10/4/18
@author: MRChou

Scenario: test cases for regression_anchors.py
"""

import tensorflow as tf

from TF_Utils.Models.FasterRCNN.regression_anchors import iou_with_gt


def iou_with_gt_test():
    # anchors in [Ymin, Xmin, Height, Width]
    anchors = tf.constant([[5.,  5.,  1.,   2.],
                           [10., 10., 20., 10.],
                           [5.,  7.,  5.,  10.]])

    # gt_boxes in [Ymin, Xmin, Ymax, Xmax]
    gt_boxes = tf.constant([[6., 6., 20., 15.],
                            [0., 0., 6.,  7.]])

    # expected answer: tensor [[0.,              2/(42-2)      ],
    #                          [50/(200+126-50), 0.            ],
    #                          [0.,              32/(50+126-32)]]
    #                        =[[0.,    0.05 ],
    #                          [0.181, 0.   ],
    #                          [0.222, 0.   ]]
    result = iou_with_gt(anchors, gt_boxes,)
    index = tf.argmax(result, axis=1)
    result = tf.reduce_max(result, axis=1, keepdims=True)
    gather = tf.gather(gt_boxes, index)
    a, b, c = tf.Session().run([index, result, gather])
    print(a)
    print(b)
    print(c)


if __name__ == '__main__':
    iou_with_gt_test()
