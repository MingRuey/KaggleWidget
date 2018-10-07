# -*- coding: utf-8 -*-
"""
Created on 9/22/18
@author: MRChou

Scenario: stores various custom loss function.
"""

import tensorflow as tf
from tensorflow import scalar_mul
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, nn, clip_ops, array_ops
from tensorflow.python.keras import backend as K

_EPSILON = 1e-7


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.
    Returns:
      A tensor.
    """
    return ops.convert_to_tensor(x, dtype=dtype)


def _weighted_categorical_crossentropy(target, output, from_logits=False, axis=1):
    """Categorical crossentropy between an output tensor and a target tensor.
    Arguments:
      target: A tensor of the same shape as `output`.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
    Returns:
      Output tensor.
    """

    fn_weight = 4

    rank = len(output.shape)
    axis = axis % rank
    # Note: nn.softmax_cross_entropy_with_logits_v2
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output = output / math_ops.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        epsilon_ = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
        return -math_ops.reduce_sum(scalar_mul(fn_weight, target) * math_ops.log(output), axis)
    else:
        return nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                       logits=output)


# keras-supported
def weighted_categorical_crossentropy(y_true, y_pred):
    return K.mean(_weighted_categorical_crossentropy(y_true, y_pred), axis=-1)


def _focal_loss(alpha, gamma):

    def loss(target, output):

        with ops.name_scope(None, "focal_loss", [target, output]) as name:
            target = ops.convert_to_tensor(target, name="target")
            output = ops.convert_to_tensor(output, name="output")

            try:
                target.get_shape().merge_with(output.get_shape())
            except ValueError:
                msg = 'focal_loss: target/output are not same shape ({} vs {})'
                msg = msg.format(target.get_shape(), output.get_shape())
                raise ValueError(msg)

        # clip output value to [_EPSILON, 1. - epsilon_],
        # prevent overflow from log(output) & log(1 - output)
        epsilon_ = _to_tensor(_EPSILON, output.dtype.base_dtype)
        output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

        # the focal loss formula is:
        # -α * (1 - σ)^𝛾 * log(σ) when y=1, -(1-α)* σ^𝛾 * log(1 - σ) when y=0
        # where α, 𝛾 are params, y: target value, σ: output(in range[0, 1])
        pos = -alpha * math_ops.pow(1-output, gamma) * math_ops.log(output)
        neg = -(1-alpha) * math_ops.pow(output, gamma) * math_ops.log1p(-output)
        loss_tensor = array_ops.where(target > 0, pos, neg)

        return math_ops.reduce_sum(loss_tensor)

    return loss


# keras-supported
def focal_loss(y_true, y_pred):
    return _focal_loss(alpha=0.1, gamma=2)(y_true, y_pred)


# tensorflow-supported
def smoothl1(x, sigma, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
        (https://arxiv.org/pdf/1504.08083v2.pdf)

                    0.5 * (sigma * x)^2         if |x| < 1/sigma^2
    smoothL1(x) = {
                    |x| - 0.5/sigma^2           otherwise
    """
    conditional = tf.less(tf.abs(x), 1 / sigma ** 2)
    close = 0.5 * (sigma * x) ** 2
    far = tf.abs(x) - 0.5 / sigma ** 2

    return tf.losses.compute_weighted_loss(tf.where(conditional, close, far),
                                           reduction=reduction)


if __name__ == '__main__':
    pass
