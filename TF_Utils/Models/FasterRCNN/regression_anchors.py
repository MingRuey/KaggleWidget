# -*- coding: utf-8 -*-
"""
Created on 10/4/18
@author: MRChou

Scenario: Helper functions for doing box regression on anchors.

          Anchors are defined by tensor of shape [n, 4], n be number of anchors.
          A row of the tensor consist of [Ymin, Xmin, Height, Width].

          Ground truth box are defined by [Ymin, Xmin, Ymax, Xmax]
"""

import tensorflow as tf

EPSILON = 1e-10

def delta_regression(delta, anchors, img_shape):
    """
    Args:
        delta: Tensor of shape [n, 4] of value [dy, dx, dh, dw]
        anchors: Tensor of same shape, of value [Ymin, Xmin, Height, Width]
        img_shape: A tuple specify the input image shape.
                   (used for clipping the output value)

    Return:
        A tuple of two tensors (Proposals tensor, Valid Proposals)

        Proposals tensor:  Tensor of shape [n, 4] of [Ymin, Xmin, Ymax, Xmax]
                           Values are clipped to in range [0, image size].
        Valid Proposals:   Boolean tensor of shape [n], it's the indices
                           that corresponding proposals has negative/small
                           height or width (True: noraml, False: negative/small)
    """
    dy, dx, dh, dw = tf.split(delta, 4, axis=1)
    y, x, h, w = tf.split(anchors, 4, axis=1)
    ymin = dy * h + y
    xmin = dx * w + x
    ymin = tf.clip_by_value(ymin, 0, img_shape[0] - 1)
    xmin = tf.clip_by_value(xmin, 0, img_shape[1] - 1)

    # for numerical stability:
    # exp(10.) ~ 2 * 10^5, is gauranteed to be clipped by than img_shape
    ymax = tf.exp(tf.minimum(dh, 10.0))*h + y
    xmax = tf.exp(tf.minimum(dw, 10.0))*w + x
    ymax = tf.clip_by_value(ymax, 1, img_shape[0] - 1)
    xmax = tf.clip_by_value(xmax, 1, img_shape[1] - 1)

    # filter index that has negative/small width or height
    is_valid = tf.logical_and((ymax - ymin) > 10.0,
                              (xmax - xmin) > 10.0)

    return tf.concat([ymin, xmin, ymax, xmax], axis=1), is_valid[:, 0]


def gtbox_to_delta(gt_boxes, anchors):
    """
    Args:
        gt_boxes: Tensor of shape [n, 4], n is the number of bounding box.
                  Each row is [Ymin, Xmin, Ymax, Xmax].
        anchors: Tensor of shape [n, 4], n is the number of anchors.
                 Each row is [Ymin, Xmin, Height, Width]

    Return:
        A corresponding tensor of shape [n, 4],
        each row is the delta [dy, dx, dw, dh] from anchors to gt_boxes.
    """
    yt, xt, yt_max, xt_max = tf.split(gt_boxes, 4, axis=1)
    ya, xa, ha, wa = tf.split(anchors, 4, axis=1)
    y_delta = (yt - ya) / ha
    x_delta = (xt - xa) / wa
    h_delta = tf.log((yt_max - yt) / ha + EPSILON)
    w_delta = tf.log((xt_max - xt) / wa + EPSILON)
    return tf.concat([y_delta, x_delta, h_delta, w_delta], axis=1)


def iou_with_gt(anchors, gt_boxes):
    """
    Credits: borrow from walkerlala's comment on
             https://gist.github.com/vierja/38f93bb8c463dce5500c0adf8648d371
    Args:
        anchors: Tensor of shape [n, 4], n is the number of anchors,
                 with each row being an anchor of [Ymin, Xmin, Height, Width].

        gt_boxes: Tensor of shape [p, 4], p is the number of groud truth boxes,
                  with each row being an box of [Ymin, Xmin, Ymax, Xmax].

    Return:
        Tensor of shape [n, p],
        with the IoU of anchors[i] and gt_boxes[j] at [i, j]
    """
    y11, x11, h11, w11 = tf.split(anchors, 4, axis=1)
    y21, x21, y22, x22 = tf.split(gt_boxes, 4, axis=1)

    y12 = y11 + h11
    x12 = x11 + w11
    h21 = y22 - y21
    w21 = x22 - x21

    xi1 = tf.maximum(x11, tf.transpose(x21))
    xi2 = tf.minimum(x12, tf.transpose(x22))

    yi1 = tf.maximum(y11, tf.transpose(y21))
    yi2 = tf.minimum(y12, tf.transpose(y22))

    inter_area = tf.maximum((xi2 - xi1), 0) * tf.maximum((yi2 - yi1), 0)

    anchors_area = w11 * h11
    gt_boxes_area = w21 * h21

    union = (anchors_area + tf.transpose(gt_boxes_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union+EPSILON)


if __name__ == '__main__':
    pass
