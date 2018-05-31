"""

Created on May 31 18:00 2018
@author: MRChou

An object allowing batch loading images from given paths.
For the general usage of trainning CNN models.

"""

import collections
import logging
from random import shuffle as rndshuffle
import numpy as np
from cv2 import imread, resize, flip, warpAffine
from cv2 import getRotationMatrix2D as getRotateMat

DATA = collections.namedtuple('DATA', ('img_path', 'label'))


class ImgLabelLoader:

    def __init__(self, imgs=(), labels=(), img_size=(300, 300, 3)):
        self.train_samples = self._form_train(imgs, labels)
        self.img_size = img_size

    def update_samples(self, imgs, labels):
        self.train_samples = self._form_train(imgs, labels)

    @staticmethod
    def _form_train(imgs, labels):
        """turn images and labels into a list of named tuples DATA"""
        samples = []
        for img in zip(imgs, labels):
            samples.append(DATA(*img))
        return samples

    def _one_epoch(self, *, shuffle=False, augment=False):
        """Generator of one epoch of train_samples"""
        samples = self.train_samples[:]
        if shuffle:
            rndshuffle(samples)

        for sample in samples:
            try:
                img = imread(sample.img_path)[..., ::-1]  # from BGR to RGB.
            except (IOError, ValueError) as err:
                logging.warning('Loading {0}: {1}'.format(sample.img_path, err))
            else:
                img = resize(img, (self.img_size[0], self.img_size[1]))

                # If augment keyword set True, yield 8x more data:
                # Rotation of 0, 90, 180 & 270 deg, along with flipped one each.
                if augment:
                    for angle in (0, 90, 180, 270):
                        for reflect in (False, True):
                            item = img.copy()
                            w, h = img.shape[0], img.shape[1]
                            if angle:
                                mat = getRotateMat((w/2, h/2),
                                                   angle=angle, scale=1.0)
                                item = warpAffine(item, mat, (w, h))
                            if reflect:
                                item = flip(item, 1)
                            yield item, sample.label
                else:
                    yield img, sample.label

    def labelshape(self):
        return self.train_samples[0].label.shape

    def batch_gener(self, batch_size, *, epoch=0, shuffle=False, augment=False):
        """A batch-image generator, if epoch <=0, it will loops infinitely"""

        assert batch_size >= 1 and batch_size % 1 == 0, \
            'Batch size must be natural numbers.'
        assert (epoch <= 0) or (epoch >= 1 and epoch % 1 == 0), \
            'Epoch must be natural numbers.'

        epoch_count = 0
        while True:
            batch_count = 0
            batch_img = np.zeros((batch_size, *self.img_size), dtype='uint8')
            batch_lab = np.zeros((batch_size,
                                  *self.train_samples[0].label.shape),
                                 dtype='uint8')

            # start collecting one batch
            for img, label in self._one_epoch(shuffle=shuffle, augment=augment):
                batch_img[batch_count, :, :, :] = img
                batch_lab[batch_count, :] = label
                batch_count += 1
                if batch_count == batch_size:
                    batch_count = 0
                    yield batch_img, batch_lab

            epoch_count += 1
            if epoch and epoch_count == epoch:
                break
