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
from keras.utils import multi_gpu_model
from keras.callbacks import Callback

DATA = collections.namedtuple('DATA', ('img_path', 'label'))


class ImgLabelLoader:
    """Used to generate (img, label) pairs for keras fit.generator"""

    def __init__(self, imgs=(), labels=(), img_size=(300, 300, 3)):
        self.num_of_samples = 0
        self.train_samples = self._form_train(imgs, labels)
        self.img_size = img_size

    def _form_train(self, imgs, labels):
        """turn images and labels into a list of named tuples DATA"""
        samples = []
        for img in zip(imgs, labels):
            samples.append(DATA(*img))

        self.num_of_samples = len(samples)
        return samples

    def update_samples(self, imgs, labels):
        self.train_samples = self._form_train(imgs, labels)

    def labelshape(self):
        return self.train_samples[0].label.shape

    def _one_epoch(self, *, shuffle=False, augment=False):
        """Generator of one epoch of train_samples"""
        samples = self.train_samples[:]
        if shuffle:
            rndshuffle(samples)

        for sample in samples:
            try:
                img = imread(sample.img_path)[..., ::-1]  # from BGR to RGB.
            except (IOError, ValueError, TypeError) as err:
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


class _TrainHistory(Callback):
    def __init__(self, model, path):
        super(Callback, self).__init__()

        self.model = model
        self.path = path
        self.best_loss = np.inf
        self.batch_losses = None
        self.epoch_val_losses = None
        self.epoch_losses = None

    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.epoch_val_losses = []
        self.epoch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_val_losses.append(logs.get('val_loss'))
        self.epoch_losses.append(logs.get('loss'))
        val_loss = logs['val_loss']

        if val_loss < self.best_loss:
            print("\nvali loss from {} to {} cover old model".format(
                self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss


class KerasModelTrainner:
    """Utility used to train keras model"""

    def __init__(self, model, model_name, train_loader, vali_loader):
        assert isinstance(train_loader, ImgLabelLoader)
        assert isinstance(vali_loader, ImgLabelLoader)
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.target = None
        self._multigpu = True
        self.optimizer = None
        self.loss = None
        self.batchsize = None
        self.epoch = None

    def compile(self, *, optimizer, loss, multi_gpu=True):
        """Compile the model with gpu setting"""
        self._multigpu = multi_gpu
        if multi_gpu:
            self.target = multi_gpu_model(self.model, gpus=2)
        else:
            self.target = self.model
        self.target.compile(optimizer=optimizer, loss=loss)
        self.target.summary()

        self.optimizer = optimizer
        self.loss = loss

    @staticmethod
    def log_config(filename):
        """The log setting during self.fit()"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=filename, level=logging.INFO
                            )

    def fit(self, *, batch_size, epoch, augment=False,
            queue=10, log='keras.log'):

        if log is not None:
            self.log_config(filename=log)

        # get batch generator from self.loader
        train_gener = self.train_loader.batch_gener(batch_size=batch_size,
                                                    epoch=epoch,
                                                    augment=augment)
        vali_gener = self.vali_loader.batch_gener(batch_size=batch_size,
                                                  epoch=epoch,
                                                  augment=augment)

        # Calculate number of steps in train and validation
        train_steps = np.floor(self.train_loader.num_of_samples / batch_size)
        if augment:
            # 8 for 0, 90, 180 and 270 deg rotation, along with flipped or not.
            train_steps *= 8
        vali_steps = np.floor(self.vali_loader.num_of_samples / batch_size)

        # Create callbacks
        history = _TrainHistory(self.model, self.model_name+'_weight.h5')

        # Now fit.
        self.target.fit_generator(steps_per_epoch=train_steps,
                                  epochs=epoch,
                                  generator=train_gener,
                                  validation_steps=vali_steps,
                                  validation_data=vali_gener,
                                  max_queue_size=queue,
                                  use_multiprocessing=True,
                                  callbacks=[history]
                                  )

        self.batchsize = batch_size
        self.epoch = epoch

        # write out log if needed
        if log:
            logging.info(history.batch_losses)
            logging.info(history.epoch_losses)
            logging.info(history.epoch_val_losses)

    def write_info(self):
        with open(self.model_name + '.info', 'w') as f:
            f.writelines(['Filename: {0}\n'.format(self.model_name + '.h5'),
                          'Model Discription:\n',
                          'Base Model:\n',
                          'Top Model:\n',
                          'Optimizer: {0} \n'.format(self.optimizer),
                          'Loss function: {0}\n'.format(self.loss),
                          'Train Data:\n',
                          'Validation Data:\n',
                          'Batchsize={0};epoch={1}\n'.format(
                              self.batchsize, self.epoch),
                          'Scores: train loss=; vali loss=\n']
                         )

    def save(self):
        """Load weights and save model"""
        if self._multigpu:
            output_model = multi_gpu_model(self.model, gpus=2)
        else:
            output_model = self.model
        output_model.load_weights(self.model_name+'_weight.h5')
        self.model.save(self.model_name + '.h5')
