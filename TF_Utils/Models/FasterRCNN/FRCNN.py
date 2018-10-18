# -*- coding: utf-8 -*-
"""
Created on 10/7/18
@author: MRChou

Scenario: Fast RCNN layers in faster RCNN, including ROI pooling.
"""

import tensorflow as tf
from tensorflow.image import crop_and_resize

from TF_Utils.Models.CustomLoss import smoothl1
from TF_Utils.Models.FasterRCNN.regression_anchors import iou_with_gt
from TF_Utils.Models.FasterRCNN.regression_anchors import gtbox_to_delta
from TF_Utils.Models.FasterRCNN.regression_anchors import delta_regression

# params
NMS_TOPN = 1024  # number of anchors selected after NMS.
MAX_NUM_POS = 32  # number of maximum positive proposals used to calculate  loss
FC_HIDDEN = [1024, 1024]
FC_DROPOUT = 0.8


class FastRCNN:

    def __init__(self, rpn, num_of_classes, is_trainning):

        self.RPN = rpn
        self.clsNum = num_of_classes
        self.isTrain = is_trainning

        self.proposals = self._get_proposals(nms_top_n=NMS_TOPN)
        self.roi = self._roi_pooling(self.proposals)
        self.fc = self._fc(self.roi)
        self.clsPred = self._cls_pred(self.fc)
        self.boxPred = self._box_pred(self.fc)

    def _roi_pooling(self, proposals):
        with tf.variable_scope('frcnn/roi'):
            # map proposals back to feature map
            # each proposal represents (row_min, col_min, row_max, col_max)
            # TODO: implicitly assusm RPN.featH = RPN.featW
            proposals = proposals / (self.RPN.anc_basesize * self.RPN.featH)

            # variant of ROI pooling: crop image to (14, 14) then max pool.
            if self.RPN.fmt == 'channels_first':
                roi = tf.transpose(self.RPN.inputs, [0, 2, 3, 1])
                roi = crop_and_resize(roi,
                                      proposals,
                                      tf.zeros(tf.shape(proposals)[0],
                                               dtype=tf.int32),
                                      crop_size=[14, 14])
                roi = tf.transpose(roi, [0, 3, 1, 2])
            else:
                roi = crop_and_resize(self.RPN.inputs,
                                      proposals,
                                      tf.zeros(tf.shape(proposals)[0],
                                               dtype=tf.int32),
                                      crop_size=[14, 14])
            roi = tf.layers.max_pooling2d(roi,
                                          pool_size=[2, 2],
                                          strides=2,
                                          padding='SAME',
                                          data_format=self.RPN.fmt)

            return roi

    def _fc(self, roi):
        with tf.variable_scope('frcnn/fc'):
            fc = tf.layers.flatten(roi)
            for hidden in FC_HIDDEN:
                fc = tf.layers.dense(fc,
                                     hidden,
                                     kernel_initializer=self.RPN.initer)
                fc = tf.layers.dropout(fc,
                                       rate=FC_DROPOUT,
                                       training=self.isTrain)
        return fc

    def _cls_pred(self, fc_layer_innput):
        with tf.variable_scope('frcnn/cls_pred'):
            return tf.layers.dense(fc_layer_innput, self.clsNum + 1)

    def _box_pred(self, fc_layer_innput):
        with tf.variable_scope('frcnn/box_pred'):
            return tf.layers.dense(fc_layer_innput, 4)

    def _get_proposals(self, nms_top_n):
        # Note that proposals are in (Ymin, Xmin, Ymax, Xmax),
        # while return values are in (Ymin, Xmin, Height, Width)
        proposals = tf.stop_gradient(self.RPN.proposals(nms_top_n=nms_top_n))
        ymin, xmin, ymax, xmax = tf.split(proposals, 4, axis=1)
        return tf.concat([ymin, xmin, ymax-ymin, xmax-xmin], axis=1)

    def _proposal_labels(self, gtcls, gtboxes, iou_thres=0.5):
        """
        Args:
            gtboxes: Tensor of shape [g, 4]
            gtcls: Tensor of shape [g, ]

            g is the number of groud truth boxes

        Return:
            Tensor of shape [p,] and [p, 4], labels and corresponding
            ground truth box repectively, with p being the number of proposals.

            Label= N: proposal labeled as N-th class,
                 = 0: proposal labeled as background.
        """
        with tf.variable_scope('proposal_labels'):
            # get the maximum IoU of each label
            ious = iou_with_gt(self.proposals, gtboxes)
            index = tf.argmax(ious, axis=1)
            gtbox = tf.gather(gtboxes, index)
            gtcls = tf.gather(gtcls, index)
            ious = tf.reduce_max(ious, axis=1)

            # thresholding on IoUs
            zeros = tf.zeros_like(ious, dtype=tf.int64)
            labels = tf.where(ious > iou_thres, gtcls, zeros)

        return labels, gtbox

    @staticmethod
    def _get_batch_index(labels, max_num_pos):
        # Choose at most max_num_pos indices from labels with non-background.
        # Then choose as many as background labels.
        # Return those indices.
        # (indices than can be used by tf.gather by the caller)
        with tf.variable_scope('batch_index'):
            index_obj = tf.where(tf.not_equal(labels, 0))
            index_obj = tf.random_shuffle(index_obj)
            index_obj = index_obj[:max_num_pos, 0]

            index_bg = tf.where(tf.equal(labels, 0))
            index_bg = tf.random_shuffle(index_bg)
            index_bg = index_bg[:tf.size(index_obj), 0]

        return index_obj, index_bg

    def loss(self, gtcls, gtboxes, batch=MAX_NUM_POS):
        """
        Args:
            gtcls: Tensor of shape [g, ], class labels for groud truth objects.
                   With value 0: background, n: n-th class.
            gtboxes: Tensor of shape [g, 4], ground truth boxes.
                     g is the number of ground truth labels.

            batch: number of positive proposals from rpn evaluated in loss.
        """
        with tf.variable_scope('frcnn/loss'):
            labels, gt_maxiou = self._proposal_labels(gtcls, gtboxes)

            # sample proposals as mini-batch with 1:1 positive negative ratio
            index_obj, index_bg = self._get_batch_index(labels, batch)

            # calculate class loss and box regression loss
            cls_index = tf.concat([index_obj, index_bg], axis=0)
            cls_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.gather(labels, cls_index),
                logits=tf.gather(self.clsPred, cls_index)
            )

            # calculate box regression loss
            delta = self.boxPred - gtbox_to_delta(gt_maxiou, self.proposals)
            box_loss = smoothl1(tf.gather(delta, index_obj), sigma=1)
            return (1/batch) * (cls_loss + box_loss)

    def predict(self):
        with tf.variable_scope('frcnn/predict'):
            pred_box, is_valid = delta_regression(self.boxPred,
                                                  self.proposals,
                                                  self.RPN.imgShape)
            pred_box = tf.boolean_mask(pred_box, is_valid)
            pred_prob = tf.boolean_mask(self.clsPred, is_valid)
            pred_prob = tf.nn.softmax(pred_prob, axis=1)
            pred_cls = tf.argmax(pred_prob, axis=1)

            # filter out background
            is_foreground = tf.not_equal(pred_cls, 0)
            pred_box = tf.boolean_mask(pred_box, is_foreground)
            pred_prob = tf.boolean_mask(pred_prob, is_foreground)
            pred_cls = tf.boolean_mask(pred_cls, is_foreground)

            # add batch dimension
            pred_box = tf.expand_dims(pred_box, axis=0)
            pred_prob = tf.expand_dims(pred_prob, axis=0)
            pred_cls = tf.expand_dims(pred_cls, axis=0)

        return {'class': pred_cls,
                'probability': pred_prob,
                'bbox': pred_box
                }


if __name__ == '__main__':
    pass
