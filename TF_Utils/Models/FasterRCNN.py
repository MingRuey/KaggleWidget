# -*- coding: utf-8 -*-
"""
Created on 9/17/18
@author: MRChou

Scenario: A self-implementation of FasterRCNN models, with tf.estimator API.
"""

import logging
import tensorflow as tf


def _anchor(row, col, base_size, scales, ratios):
    """Generate n anchors for cell at row, col in feature map,
       where n = anc_sizes x anz_ratios

    Args:
        col: the column location of the cell.
        row: the row location of the cell.

             note that top-left most cell has (row, col) = (0, 0)

        base_size: the ratio between input image and feature map.

        scales: the size of anchors in pixels
        ratios: the width/hight ratios of anchors.

    Return:

        Anchors array of shape: [1, 1, 1, 1, n] => [ymin, ymax, xmin, xmax, n],
        xmin, xmax, ymin, ymax are the pixel locations of anchor.
    """

    center = tf.constant([base_size*(col + 0.5), base_size*(row + 0.5)])

    def single_anchor(ctr_col, ctr_row, scale, ratio):
        anchor = tf.constant([])

    base_anchor = tf.constant([0, 0, base_size, base_size])

    first_anchor = tf.constant([0, 0, base_size*scales[0], base_size*scales[0]])


def _rpn(inputs, data_format,
         rpn_channels=512, anc_sizes=(8, 16, 32), anc_ratios=(0.5, 1, 2),
         initializer=tf.variance_scaling_initializer()):
    """regional proposal network

    Args:
        inputs: should be feature maps from some CNN
        data_format: the input format ('channels_last' or 'channels_first').

        rpn_channels : number of filters for the RPN layer
        anc_sizes : a list of integers to specify sizes of anchors
        anc_ratios : a list of floats to specify aspect ratios of anchor.

        initializer : the initializer for setting up weights.
    """

    with tf.variable_scope('rpn'):
        if data_format == 'channel_first':
            pad = [[0, 0], [0, 0], [1, 1], [1, 1]]
        else:
            pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        windows = tf.pad(inputs, paddings=pad, name='sliding_windows_pad')
        windows = tf.layers.conv2d(
            inputs=windows, kernel_size=3, strides=1, filters=rpn_channels,
            activation=tf.nn.relu, kernel_initializer=initializer,
            data_format=data_format, name='sliding_windows_conv')

        cls_score = tf.layers.conv2d(
            inputs=windows, kernel_size=1, strides=1,
            filters=len(anc_sizes)*len(anc_ratios)*2,
            kernel_initializer=initializer,
            data_format=data_format, name='cls_score')


if __name__ == '__main__':
    # model = tf.keras.applications.ResNet50(include_top=False, pooling='avg',
    #                                        weights=None)
    # print(model.summary())
    pass
