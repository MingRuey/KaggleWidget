# -*- coding: utf-8 -*-
"""
Created on Apr. 29. 2018.
@author: cchsia
@author: mrchou

define different model used in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""
import os
import logging
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from keras.models import load_model
import numpy as np
from iMfashion_ImgBatchLoader import ImgBatchLoader

# model.save() cause some problem when using multi GPU
# Checkpoint use model.save() ...

class _train_history(Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.epoch_val_losses = []
        self.epoch_losses = []
    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_val_losses.append(logs.get('val_loss'))
        self.epoch_losses.append(logs.get('loss'))
        val_loss = logs['val_loss']
        # self.model.save_weights("test_v3_drop_{}.h5".format(epoch))
        if val_loss < self.best_loss:
            print("\nvali loss from {} to {} cover old model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss


def model_continue(model_path):
    model = load_model(model_path, compile=False)
    return model


def model_IncepV3():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # flatten vs avgPooling: https://github.com/keras-team/keras/issues/8470
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(228, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def model_IncepV3_withDrop():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    predictions = Dense(228, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def model_DenseNet169():
    base_model = DenseNet169(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(228, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def model_ResNet50():
    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(228, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def model_VGG16():
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(228, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


class model_trainner:

    def __init__(self, model, model_name, train_path, train_label, vali_path, vali_label):
        self.model = model
        self.model_name = model_name
        self.train_path = train_path
        self.train_label = train_label
        self.vali_path = vali_path
        self.vali_label = vali_label

    def log_config(self):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename='model_trainner.log', level=logging.INFO
                            )

    def fit(self, optimizer, loss, batch_size, epoch, augmenting=False, multi_gpu=2, workers=16, queue=10, log=True):
        assert batch_size >= 1 and batch_size%1==0, 'Batchsize must be natural numbers.'
        assert epoch>=1 and epoch%1==0, 'Epoch must be natural numbers.'
        if log:
            self.log_config()

        # create images generator
        imgs_train = ImgBatchLoader(img_path=self.train_path, img_label=self.train_label)
        imgs_vali = ImgBatchLoader(img_path=self.vali_path, img_label=self.vali_label)

        # compile model with gpu setting
        target = multi_gpu_model(self.model, gpus=multi_gpu) if multi_gpu else self.model
        target.compile(optimizer=optimizer, loss=loss)
        target.summary()

        history = _train_history(self.model, self.model_name+'_weight.h5')
        train_steps = len([i for i in os.listdir(self.train_path) if i.lower().endswith('.jpg')])/batch_size
        vali_steps = len([i for i in os.listdir(self.vali_path) if i.lower().endswith('.jpg')])/batch_size

        if augmenting:
            train_steps *= 8  # rotate with 0, 90, 180 and 270 degree, could also being flipped or not.

        target.fit_generator(validation_steps=vali_steps,
                            steps_per_epoch=train_steps, epochs=epoch,
                            generator=imgs_train.generator(batch_size, augmenting=augmenting),
                            validation_data=imgs_vali.generator(batch_size),
                            use_multiprocessing=True,
                            workers=workers,
                            max_queue_size=queue,
                            callbacks=[history]
                            )

        # load weight and save model
        output_model = multi_gpu_model(self.model, gpus=multi_gpu) if multi_gpu else self.model
        # output_model.load_weights(self.model_name + '_weight.h5')
        output_model.load_weights(self.model_name+'_weight.h5')
        self.model.save(self.model_name + '.h5')

        fw = open(self.model_name + '.info', 'w')
        fw.writelines(['Filename: {0}   \n'.format(self.model_name + '.h5'), \
                       'Model Discription: \n', \
                       'Base Model: \n',\
                       'Top Model:  \n',\
                       'Optimizer: {0}  \n'.format(optimizer),\
                       'Loss function: {0} \n'.format(loss),\
                       'Train Data: {0} \n'.format(self.train_path),\
                       'Validation Data: {0} \n'.format(self.vali_path),\
                       'Fit: batchsize={0}; epoch={1} \n'.format(batch_size, epoch), \
                       'Scores: train loss=; vali loss= \n']
                      )
        fw.close()

        if log:
            logging.info(history.batch_losses)
            logging.info(history.epoch_losses)
            logging.info(history.epoch_val_losses)


def main():

    train_path = '/rawdata/FGVC5_iMfashion/imgs_train/'
    train_label = '/archive/iMfashion/labels/labels_train.pickle'
    vali_path = '/rawdata/FGVC5_iMfashion/imgs_validation/'
    vali_label = '/archive/iMfashion/labels/labels_validation.pickle'

    s = model_trainner(model=model_continue('/archive/iMfashion/models/IncepV3+drop_0513_iM5.h5'),
                       # model=model_IncepV3_withDrop(),
                       model_name='IncepV3+drop_0514_iM6',
                       train_path=train_path,
                       train_label=train_label,
                       vali_path=vali_path,
                       vali_label=vali_label
                       )

    s.fit(optimizer='rmsprop',
          loss='binary_crossentropy',
          batch_size=128,
          epoch=10,
          multi_gpu=2,
          log=True
          )

    # summarize history for loss
    def plot_history(fit_history):
        import matplotlib.pyplot as plt
        # list all data in history
        # print(fit_history.history.keys())
        # plt.plot(fit_history.history['loss'])
        # plt.plot(fit_history.history['val_loss'])
        plt.plot(fit_history.epoch_losses)
        plt.plot(fit_history.epoch_val_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('lost.png')

    # plot training process
    # plot_history(history)


if __name__ == '__main__':
    main()
