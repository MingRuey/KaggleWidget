# -*- coding: utf-8 -*-
"""
Created on 10/4/18
@author: MRChou

Scenario: Regional proposal network in Faster RCNN.
"""

import tensorflow as tf
from tensorflow.image import non_max_suppression

from TF_Utils.Models.CustomLoss import smoothl1
from TF_Utils.Models.FasterRCNN.regression_anchors import iou_with_gt
from TF_Utils.Models.FasterRCNN.regression_anchors import gtbox_to_delta
from TF_Utils.Models.FasterRCNN.regression_anchors import delta_regression
from TF_Utils.Models.FasterRCNN.generate_anchors import generate_anchors
from TF_Utils.Models.FasterRCNN.generate_anchors import anchor_board_check

# params initialized when RPN.__init__(...), see __init__ doc string.
CHANNEL = 'channels_first'
RPN_CHNS = 512
ANC_SIZES = (2, 4, 8, 10, 12)
ANC_RATIOS = (0.5, 1, 2)

# params when RPN.someMthd called
IOU_THRES = 0.5  # threshold for determining positive/negative anchors
MINIBATCH = 256  # number of anchors used to calculate RPN loss
BOXLOSSWEIGHT = 10  # for weighting between cls_loss and box_loss


class RPN:
    # TODO: Support non-square img. Img and feature map are assumed square now.
    """
    Args:
        inputs: the input tensor, shuoud be feature map of some CNN.
        img_shape: (height, width) shape of input image
        datafmt: specify 'channels_first' or 'channels_last' of inputs.
        rpn_channels: the number of channels of sliding window convolution.
        anc_sizes: a list of integers to specify sizes of anchors.
        anc_ratios: a list of floats to specify aspect ratios of anchor.
    """

    def __init__(self, inputs, img_shape,
                 datafmt=CHANNEL, rpn_channels=RPN_CHNS,
                 anc_sizes=ANC_SIZES, anc_ratios=ANC_RATIOS,
                 weights_initializer=tf.variance_scaling_initializer()):
        # TODO: only sinlge batch supported now.
        self.inputs = inputs
        self.fmt = datafmt
        if self.fmt == 'channels_first':
            self.featH = int(inputs.get_shape()[2])
            self.featW = int(inputs.get_shape()[3])
        else:
            self.featH = int(inputs.get_shape()[1])
            self.featW = int(inputs.get_shape()[2])
        self.featNum = self.featH*self.featW
        self.imgShape = img_shape
        self.anc_basesize = self.imgShape[0] / self.featH

        self.rpnChns = rpn_channels
        self.ancSizes = anc_sizes
        self.ancRatios = anc_ratios
        self.ancNum = len(anc_sizes) * len(anc_ratios)

        self.initer = weights_initializer

        self.windows = self._sliding_window_conv(self.inputs)
        self.clsProb = self._cls_prob(self.windows)
        self.boxPred = self._box_pred(self.windows)
        self.anchors, self._validAnchors = self._get_anchors()

    def _sliding_window_conv(self, inputs):
        with tf.variable_scope('rpn/sliding_window'):
            if self.fmt == 'channels_first':
                pad = [[0, 0], [0, 0], [1, 1], [1, 1]]
            else:
                pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
            windows = tf.pad(inputs, paddings=pad)
            windows = tf.layers.conv2d(windows,
                                       kernel_size=3,
                                       strides=1,
                                       filters=self.rpnChns,
                                       activation=tf.nn.relu,
                                       kernel_initializer=self.initer,
                                       data_format=self.fmt)
        return windows

    def _cls_prob(self, sliding_window_conv_outputs):
        """Return foregroud/background probabilities for each anchor"""
        with tf.variable_scope('rpn/cls_prob'):
            cls_conv = tf.layers.conv2d(sliding_window_conv_outputs,
                                        kernel_size=1,
                                        strides=1,
                                        filters=self.ancNum,
                                        kernel_initializer=self.initer,
                                        data_format=self.fmt)

            # Checkout 'change_shape_note' at bottom of class
            if self.fmt == 'channels_first':
                cls_conv = tf.transpose(cls_conv, [0, 2, 3, 1])
            cls_conv = tf.reshape(cls_conv, shape=[self.featNum*self.ancNum])
        return cls_conv

    def _box_pred(self, sliding_window_conv_outputs):
        """Return delta: [dy, dx, dh, dw] of box regression for each anchor"""
        with tf.variable_scope('rpn/box_pred'):
            box_conv = tf.layers.conv2d(sliding_window_conv_outputs,
                                        kernel_size=1,
                                        strides=1,
                                        filters=self.ancNum*4,
                                        kernel_initializer=self.initer,
                                        data_format=self.fmt)

            # Checkout 'change_shape_note' at bottom of class
            if self.fmt == 'channels_first':
                box_conv = tf.transpose(box_conv, [0, 2, 3, 1])
            box_conv = tf.reshape(box_conv, shape=[self.featNum*self.ancNum, 4])
        return box_conv

    def _get_anchors(self):
        with tf.variable_scope('rpn/generate_anchors'):
            anchors = generate_anchors(self.featW,
                                       self.featH,
                                       self.anc_basesize,
                                       self.ancSizes,
                                       self.ancRatios)
            with tf.variable_scope('anchor_boarder_check'):
                is_valid = anchor_board_check(anchors, self.imgShape)
        return anchors, is_valid

    def _anc_labels(self, gtboxes, iou_thres=IOU_THRES):
        """
        Return anchor labels based on IoU of the anchor and ground truth.
        Label=1: positive anchors,=0: negative anchors, =-1: ignored.
        """
        with tf.variable_scope('anchor_labels'):
            # get the maximum IoU of each label
            ious = iou_with_gt(self.anchors, gtboxes)
            gtbox = tf.gather(gtboxes, tf.argmax(ious, axis=1))
            ious = tf.reduce_max(ious, axis=1)

            # thresholding on IoUs
            _ones = tf.ones_like(ious)
            _zero = tf.zeros_like(ious)
            labels = tf.where(ious > iou_thres, _ones, -1*_ones)
            labels = tf.where(ious < 0.1, _zero, labels)

            # thresholding on anchor boarders
            labels = tf.where(self._validAnchors, labels, -1*_ones)

        return labels, gtbox

    @staticmethod
    def _get_batch_index(labels, minibatch):
        # Randomly choose 0.5*minibatch indices from labels with value 1,
        #             and 0.5*minibatch indicesfrom labels with value 0.
        # Return those indices.
        # (indices than can be used by tf.gather by the caller)
        with tf.variable_scope('batch_index'):
            index0 = tf.where(tf.equal(labels, 0))
            index0 = tf.random_shuffle(index0)
            index1 = tf.where(tf.equal(labels, 1))
            index1 = tf.random_shuffle(index1)

            # Size check. if there are not enough labels with value 0/1,
            #             replace the remaining with 1/0 labels.
            size0 = tf.shape(index0)[0]
            size1 = tf.shape(index1)[0]
            num = minibatch // 2

            index1 = tf.case([(size0 < num, lambda: index1[:2*num-size0, 0]),
                              (size1 < num, lambda: index1[:size1, 0])],
                             default=lambda: index1[:num]
                             )
            index0 = tf.case([(size1 < num, lambda: index0[:2*num-size1, 0]),
                              (size0 < num, lambda: index0[:size0, 0])],
                             default=lambda: index0[:num]
                             )
        return index1, index0

    def loss(self, gtboxes, batch=MINIBATCH, weight=BOXLOSSWEIGHT):
        # img_gt_boxes: Tensor of shape[g, 4], g is the number of groud truth
        #               boxes, with each row being [Ymin, Xmin, Ymax, Xmax].
        # !should match with func iou_with_gt
        with tf.variable_scope('rpn/loss'):
            labels, gt_maxiou = self._anc_labels(gtboxes)

            # sample 256 anchors as mini-batch with 1:1 positive negative ratio
            index_pos, index_neg = self._get_batch_index(labels, batch)

            # calculate class loss and box regression loss
            cls_index = tf.concat([index_pos, index_neg], axis=0)
            cls_loss = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.gather(labels, cls_index),
                logits=tf.gather(self.clsProb, cls_index)
            )

            # calculate box regression loss
            delta = self.boxPred - gtbox_to_delta(gt_maxiou, self.anchors)
            box_loss = smoothl1(tf.gather(delta, index_pos), sigma=1)

            return (1/batch)*cls_loss + (weight/self.featNum)*box_loss

    def proposals(self, nms_top_n, nms_thres=0.6):
        # Return:
        #       Tensor of shape [nms_top_n, 4],
        #       each row being [Ymin, Xmin, Ymax, Xmax] on input image scale.
        with tf.variable_scope('rpn/proposals'):
            proposals, is_valid = delta_regression(self.boxPred,
                                                   self.anchors,
                                                   self.imgShape)

            proposals = tf.boolean_mask(proposals, is_valid)
            prob = tf.boolean_mask(self.clsProb, is_valid)

            proposals = tf.gather(proposals,
                                  non_max_suppression(proposals,
                                                      prob,
                                                      score_threshold=-10.0**10,
                                                      iou_threshold=nms_thres,
                                                      max_output_size=nms_top_n
                                                      )
                                  )
        return proposals

    # "change_shape_note":
    #
    # Change dimesion of convs from [batch, A, H, W] (channels_first)
    #                            or [batch, H, W, A] (channels_last)
    #                          into [HxWxA, n], n=1 or n=4 for cls or box.
    # H, W, A --- feature map height, feature map width, number of anchors.
    #
    # After the change, we map anchors to each score easily ---
    # The resulted tensor T maps to feature map location by:
    # T[     0:      A, :] => A anchors on (0, 0) of feature map (row, col).
    # T[     A:     2A, :] => A anchors on (0, 1)
    # ...
    # T[(W-1)A:     WA, :] =>    ...    on (0, W)
    # T[    WA: (W+1)A, :] =>    ...    on (1, 0)
    # ...
    # .etc


if __name__ == '__main__':
    pass
