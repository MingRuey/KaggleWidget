# -*- coding: utf-8 -*-
"""
Created on 10/4/18
@author: MRChou

Scenario: Helper functions for generating anchors w.r.t. a feature map.
"""

from functools import partial

import tensorflow as tf


def _get_anchors_by_cell(row, col, base_size, scales, ratios):
    """Generate n anchors for cell at row, col in feature map,
       where n = anc_sizes x anz_ratios

    Args:
        row: the row location of the cell.
        col: the column location of the cell.

             note that top-left most cell has (row, col) = (0, 0)

        base_size: the ratio between input image and feature map.

        scales: tensor constant, the size of anchors in pixels
        ratios: tensor constant, the width/hight ratios of anchors.

    Return:

        Anchors array of shape: [n, 4] => n x [Ymin, Xmin, Height, Width],
        Ycenter, Xcenter, height, width are the pixel locations of anchor.
    """
    # ctr_mat is a matrix of shape [n, 2]:
    # [[Ymin, Xmin],
    #  [Ymin, Xmin],
    #  [Ymin, Xmin], ... repeat n times.
    scales = tf.constant(scales, dtype=tf.float32)
    ratios = tf.constant(ratios, dtype=tf.float32)

    num = tf.size(scales) * tf.size(ratios)
    ymin = base_size * row
    xmin = base_size * col
    corner_mat = tf.concat([tf.fill([num, 1], ymin),
                            tf.fill([num, 1], xmin)], axis=1)

    # hw_mat is a matrix of shape [n, 2]:
    # [[α1*h1, α1*w1],
    #  [α1*h2, α1*w2],
    #  ...
    #  [α1*hk, α1*wk],
    #  [α2*h1, α2*w1],
    #  [α2*h2, α2*w2],
    #  ...
    #  [αn*hk, αn*wk]]
    #
    #  where αi is the i-th number in scales.
    #        hn, wn correpond the n-th number in ratios [r1, r2, ..., rn],
    #               with hn = (1/rn) ** 0.5, wn = rn ** 0.5
    h_vec = tf.sqrt(1/ratios)
    w_vec = tf.sqrt(ratios)
    h_vec = tf.reshape(tf.tensordot(scales, h_vec, axes=0), shape=[num, 1])
    w_vec = tf.reshape(tf.tensordot(scales, w_vec, axes=0), shape=[num, 1])
    hw_mat = tf.concat([h_vec, w_vec], axis=1)

    return tf.concat([corner_mat, hw_mat], axis=1)


def generate_anchors(feat_w, feat_h, basesize, scales, ratios):
    """
    Args:
        feat_w: width W of feature map.
        feat_h: height H of feature map.
        basesize: the ratio between feature map size and input image.
        scales: a list of integers to specify sizes of anchors.
        ratios: a list of floats to specify aspect ratios of anchor.

    Return:
        Tensor of shape [featW * featH * N, 4],
               each row is an anchor of value [Ymin, Xmin, Height, Width],
               where N is the total number of anchors.

        Note that the resulted tensor T maps to feature map location by:
        T[0, :] = (0, 0) of feature map (row, col).
        T[1, :] = (0, 1) of feature map (row, col).
        ...
        T[W+1, :] = (1, 0). W is the width of feature map.
        .etc
    """
    anchors_fn = partial(_get_anchors_by_cell,
                         base_size=basesize,
                         scales=scales,
                         ratios=ratios)
    anchors = tf.concat([anchors_fn(row, col)
                         for col in range(feat_w)
                         for row in range(feat_h)], axis=0)
    return anchors


def anchor_board_check(anchors, img_shape):
    """
    Args:
        anchors: Tensor of shape [n, 4], n is the number of anchors.
                 Each row is an anchor of value [Ymin, Xmin, Height, Width].
        img_shape: Tuple specify input image shape.

    Return:
        Boolean tensor of shape [n, 1], with each element is whether the anchor
        is inside image pixel range.
    """
    row, col, height, width = tf.split(anchors, 4, axis=1)
    index_ge_0 = tf.logical_and(row > 0, col > 0)
    inside_img = tf.logical_and((row + height) < img_shape[0],
                                (col + width) < img_shape[1])
    return tf.logical_and(index_ge_0, inside_img)


if __name__ == '__main__':
    pass
