# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 07:00:00 2018
@author: MRChou

Read images in the given dir, and output a batch data via self.batch as a generator.
A tool for used by the VGG16 model, for iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""

import numpy
import os.path
import cv2
import pickle
import logging
from random import shuffle as rndshuffle


class ImgBatchLoader:

    def __init__(self, img_path, img_label, img_size=(300, 300, 3)):
        self._path = img_path
        try:
            self._labels = pickle.load(open(os.path.join(self._path, img_label), 'rb'))
        except (OSError, TypeError) as err:
            raise OSError('img_label should be a pickle file available:', err)
        self._size = img_size
        self._imgs = None

    def __load_img(self, shuffle=True, augmenting=False):
        """Generator for loading and resizing images"""
        files = os.listdir(self._path)
        if shuffle:
            rndshuffle(files)

        for file in files:
            if file.lower().endswith('.jpg'):
                imgid = int(file[:-4])
                try:
                    img = cv2.imread(os.path.join(self._path, file))[..., ::-1]  # BGR by default, convert it into RGB
                    img = cv2.resize(img, (self._size[0], self._size[1]))
                    label = self._labels[numpy.where(self._labels[:, 0] == imgid)][0, 1:]

                    # If augmenting keyword set True, yield 8x more data.
                    # Including rotation with degree 0, 90, 180 and 270, and a flipped one each.
                    if augmenting:
                        for angle in (0, 90, 180, 270):
                            for reflect in (False, True):
                                s = img.copy()
                                w, h = img.shape[0], img.shape[1]
                                if angle:
                                    m = cv2.getRotationMatrix2D((w/2, h/2), angle=angle, scale=1.0)
                                    s = cv2.warpAffine(s, m, (w, h))
                                if reflect:
                                    s = cv2.flip(s, 1)
                                yield s, label

                    yield img, label
                except (IOError, ValueError) as err:
                    logging.warning('While loading {0}: {1}'.format(file, err))
                except IndexError as err:
                    logging.warning('While finding labels for img id {0} : {1}'.format(imgid, err))

    def generator(self, batch_size, shuffle=True, epoch=None, avgbatch=False, augmenting=False):
        """A batch-data generator, if epoch not set, it will loops infinitely"""
        assert batch_size >= 1 and batch_size % 1 == 0, 'Batch size must be natural numbers.'
        assert (not epoch) or (epoch >= 1 and epoch % 1 == 0), 'Epoch must be natural numbers.'

        epoch_count = 0
        while True:
            # begining of an epoch: loading imgs
            self._imgs = self.__load_img(shuffle=shuffle, augmenting=augmenting)

            # start collecting one batch
            batch_count = 0
            img_batch = numpy.zeros((batch_size, *self._size), dtype='uint8')
            label_batch = numpy.zeros((batch_size, self._labels.shape[1]-1), dtype='uint8')
            for img, imglabel in self._imgs:
                img_batch[batch_count, :, :, :] = img
                label_batch[batch_count, :] = imglabel
                batch_count += 1
                if batch_count == batch_size:
                    batch_count = 0
                    if avgbatch:  # subtract mean if necessary
                        mean = numpy.mean(img_batch, axis=(0, 1, 2))
                        img_batch[:, :, :, 0] -= mean[0]
                        img_batch[:, :, :, 1] -= mean[1]
                        img_batch[:, :, :, 2] -= mean[2]
                    yield img_batch, label_batch

            epoch_count += 1
            if epoch and epoch_count == epoch:
                break
