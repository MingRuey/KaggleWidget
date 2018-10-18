# -*- coding: utf-8 -*-
"""
Created on 10/16/18
@author: MRChou

Scenario: Tensorflow implementation of Unet: https://arxiv.org/abs/1505.04597
          Credits:
          Mostly borrow from https://github.com/kkweon/UNet-in-Tensorflow
"""

import tensorflow as tf

EPSILON = 1e-10


def conv_pool(inputs, n_filters, training, pool=True, reg_strength=0.1):
    """
    {Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        inputs (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        pool (bool): If True, MaxPool2D

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = inputs
    with tf.name_scope("ConvPool"):
        for num in n_filters:
            net = tf.layers.conv2d(net, num, (3, 3),
                                   activation=None,
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_strength),
                                   )
            net = tf.layers.batch_normalization(net, training=training)
            net = tf.nn.relu(net)

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2))
        return net, pool


def upconv_2d(inputs, n_filter, reg_strength=0.1):
    """
    Up Convolution `tensor` by 2 times

    Args:
        inputs (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size

    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        inputs,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_strength))


# TODO: support different channel format
def unet(inputs, training, reg=0.1):
    """
    Build a U-Net architecture

    Args:
        inputs (4-D Tensor): (N, H, W, C), should be normalized into [-1, 1]
        training (1-D Tensor): Boolean Tensor is required for batchnorm
        reg: float32, strength of l2 regularization

    Returns:
        output (4-D Tensor): (N, H, W, C) Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = inputs

    # down sample process
    with tf.variable_scope('DownConv'):
        conv1, pool1 = conv_pool(net, [64, 64], training, reg_strength=reg)
        conv2, pool2 = conv_pool(pool1, [128, 128], training, reg_strength=reg)
        conv3, pool3 = conv_pool(pool2, [256, 256], training, reg_strength=reg)
        conv4, pool4 = conv_pool(pool3, [512, 512], training, reg_strength=reg)
        conv5 = conv_pool(pool4, [1024, 1024], training, reg_strength=reg, pool=False)

    # up sample process
    with tf.variable_scope('UpConvConcat'):
        up6 = upconv_2d(conv5, 512, reg_strength=reg)
        up6 = tf.concat([up6, conv4], axis=-1)
        conv6 = conv_pool(up6, [512, 512], training, reg_strength=reg, pool=False)

        up7 = upconv_2d(conv6, 256, reg_strength=reg)
        up7 = tf.concat([up7, conv3], axis=-1)
        conv7 = conv_pool(up7, [256, 256], training, reg_strength=reg, pool=False)

        up8 = upconv_2d(conv7, 128, reg_strength=reg)
        up8 = tf.concat([up8, conv2], axis=-1)
        conv8 = conv_pool(up8, [128, 128], training, reg_strength=reg, pool=False)

        up9 = upconv_2d(conv8, 64, reg_strength=reg)
        up9 = tf.concat([up9, conv1], axis=-1)
        conv9 = conv_pool(up9, [64, 64], training, reg_strength=reg, pool=False)

    return tf.layers.conv2d(conv9, 1, (1, 1),
                            activation=tf.nn.sigmoid,
                            padding='same',
                            name='Outputs')


def product_iou(y_pred, y_true):
    """

    Returns a (approx) IOU score:

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + epsilon) + epsilon

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """

    intersection = 2 * tf.reduce_sum(y_pred * y_true) + EPSILON
    denominator = tf.reduce_sum(y_pred + y_true) + EPSILON
    return tf.reduce_mean(intersection / denominator)


if __name__ == '__main__':
    pass
