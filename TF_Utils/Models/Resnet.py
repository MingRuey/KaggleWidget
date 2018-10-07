# -*- coding: utf-8 -*-
"""
Created on 9/23/18
@author: MRChou

Scenario: For building a resent model to Tensorflow Esitmator API.
          Modified from tensorflow official models:
          https://github.com/tensorflow/models/tree/master/official/resnet
"""

import tensorflow as tf

_CHANNEL = 'channels_first'  # or channels_last, see class ResnetV2
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def _block_v2(inputs, filters, training, projection_shortcut, strides,
              data_format):
    """A single block for ResNet v2, without a bottleneck.

    Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
    """A single block for ResNet v2, without a bottleneck.

    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, blocks, bottleneck, block_fn, strides, base_filters_num,
                training, name, data_format):
    """Creates one layer of blocks for the ResNet model.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block layer.
    """

    def projection_shortcut(inputs):

        # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = base_filters_num * 4 if bottleneck else base_filters_num

        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    with tf.variable_scope(name):
        # Only the first block per block_layer
        # uses projection shortcut and strides
        inputs = block_fn(inputs,
                          filters=base_filters_num,
                          strides=strides,
                          projection_shortcut=projection_shortcut,
                          training=training,
                          data_format=data_format)

        for _ in range(1, blocks):
            inputs = block_fn(inputs,
                              filters=base_filters_num,
                              strides=1,
                              projection_shortcut=None,
                              training=training,
                              data_format=data_format)

    return inputs


class ResnetV2:
    """Used for creating resnet model_fn.

    Args:
        blocks: A list containing n values, where values are the number of
                block layers desired.

        block_strides: A list containing n values, where values are the strides
                       for each bottleneck block.

        bottleneck: where to use bottleneck block or usual resnet block.

        datafmt: 'channel_first'(default) or 'channel_last'
                 # This may provide a large performance boost on GPU.
                 # www.tensorflow.org/performance/performance_guide#data_formats

        scope: name scope for model
    """

    def __init__(self, blocks, block_strides, bottleneck=True,
                 datafmt=_CHANNEL, scope='ResnetV2'):

        self.blocks = blocks
        self.block_strides = block_strides
        self.bottleneck = bottleneck
        self.block_fn = _bottleneck_block_v2 if bottleneck else _block_v2
        self.datafmt = datafmt
        self.scope = scope

    def __call__(self, inputs, istraining, pooling='avg'):
        assert self.datafmt in ('channels_first', 'channels_last')

        with tf.variable_scope(self.scope, tf.float32):
            if self.datafmt == 'channels_first':
                # from channels_last (NHWC) to channels_first (NCHW).
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # initial conv
            inputs = conv2d_fixed_padding(inputs=inputs,
                                          filters=64, kernel_size=7, strides=2,
                                          data_format=self.datafmt)
            inputs = tf.identity(inputs, 'initial_conv')

            # initial max pool
            inputs = tf.layers.max_pooling2d(inputs=inputs,
                                             pool_size=3, strides=2,
                                             padding='SAME',
                                             data_format=self.datafmt)
            inputs = tf.identity(inputs, 'initial_max_pool')

            # resnet-blocks
            for i, num_blocks in enumerate(self.blocks):
                num_filters = 64 * (2 ** i)
                inputs = block_layer(inputs=inputs,
                                     blocks=num_blocks,
                                     bottleneck=self.bottleneck,
                                     block_fn=self.block_fn,
                                     base_filters_num=num_filters,
                                     strides=self.block_strides[i],
                                     training=istraining,
                                     name='block_layer{}'.format(i + 1),
                                     data_format=self.datafmt)

            inputs = batch_norm(inputs, istraining, self.datafmt)
            inputs = tf.nn.relu(inputs)

            if pooling:
                # Final pooling layer
                axes = [2, 3] if self.datafmt == 'channels_first' else [1, 2]
                inputs = tf.reduce_mean(inputs, axes, keepdims=True)
                inputs = tf.identity(inputs, 'final_reduce_mean')

                # for resize, check: https://stackoverflow.com/questions/49547435
                final_dim = tf.reduce_prod(inputs.get_shape().as_list()[1:])
                inputs = tf.reshape(inputs, [-1, final_dim])

            return inputs


if __name__ == '__main__':
    pass
