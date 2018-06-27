# -*- coding: utf-8 -*-
"""
Created on Thu June 25 07:41 2018
@author: MRChou

Code for using trees to transform features.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import os
import pickle
import numpy
import threading
import cv2
import pandas
from math import ceil
from multiprocessing import Queue
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

TRAIN_PATH = '/rawdata/Avito_Demand/imgs_train/'
PKL_PATH = '/archive/Avito/data_preprocess/'
TRAIN_PKL = 'train.pickle'
WEIGHT_PATH = '/home/mrchou/code/KaggleWidget/Avito_Demand/weights/'


# from github.com/titu1994/neural-image-assessment/utils/score_utils.py
def mean_score(scores):
    si = numpy.arange(1, 11, 1)
    mean = numpy.sum(scores * si)
    return mean


# from github.com/titu1994/neural-image-assessment/utils/score_utils.py
def std_score(scores):
    si = numpy.arange(1, 11, 1)
    mean = mean_score(scores)
    std = numpy.sqrt(numpy.sum(((si - mean) ** 2) * scores))
    return std


def img_loader(imgs, imgids):
    img_mat = numpy.zeros((len(imgids), 224, 224, 3))
    for index, img in enumerate(imgs[imgids]):
        try:
            img = load_img(TRAIN_PATH + img + '.jpg',
                           target_size=(224, 224)
                           )
        except (OSError, TypeError) as err:
            if not pandas.isnull(img):
                print('Error when loading', img, err)
        else:
            img = img_to_array(img)
            img = numpy.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img_mat[index, :] = img
    return img_mat


# NIMA score
#  - original paper: https://arxiv.org/abs/1709.05424
#  - realization from: https://github.com/titu1994/neural-image-assessment/
def nima(imgs, batch_size=1024):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False,
                           pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(WEIGHT_PATH + 'mobilenet_weights.h5')

    mat_out = numpy.zeros((len(imgs), 10))
    total_batch = ceil(len(imgs)/batch_size)
    for n in range(total_batch):
        print('Currently {} th batch out of {}.'.format(n, total_batch))
        if (n + 1)*batch_size > len(imgs):
            imgids = [i for i in range(n * batch_size, len(imgs))]
        else:
            imgids = [i for i in range(n * batch_size, (n+1) * batch_size)]

        img_mat = img_loader(imgs, imgids)
        mat_out[imgids, :] = model.predict(img_mat, batch_size=batch_size)
    return mat_out


def brightness(img):
    return img.sum()/img.size


def key_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    return kp


class ImgWorker(threading.Thread):

    def __init__(self, in_que, out_que):
        super(ImgWorker, self).__init__()
        self._in_que = in_que
        self._out_que = out_que

    def run(self):
        while True:
            task = self._in_que.get()
            if task == 'done':
                break
            else:
                index, img = task

                try:
                    img = cv2.imread(TRAIN_PATH + img + '.jpg')
                except (OSError, TypeError) as err:
                    if not pandas.isnull(img):
                        print('Error when loading', img, err)
                else:
                    self._out_que.put((index,
                                       brightness(img),
                                       len(key_points(img))
                                       )
                                      )


def build_proc(in_que, out_que, num_of_workers=16):
    workers = []
    for _ in range(num_of_workers):
        worker = ImgWorker(in_que, out_que)
        worker.start()
        workers.append(worker)
    return workers


def script_nima():
    with open(os.path.join(PKL_PATH, TRAIN_PKL), 'rb') as f:
        with open(os.path.join(PKL_PATH, 'nima.pickle'), 'wb') as f_out:
            df_train = pickle.load(f)
            pickle.dump(nima(df_train['image'].values), f_out)


def script_brightness():
    with open(os.path.join(PKL_PATH, TRAIN_PKL), 'rb') as f:
        with open(os.path.join(PKL_PATH, 'bright_kp.pickle'), 'wb') as f_out:
            df_train = pickle.load(f)
            imgs = df_train['image'].values

            q_tasks = Queue()
            q_results = Queue()
            processers = build_proc(in_que=q_tasks, out_que=q_results)

            for index, img in enumerate(imgs):
                q_tasks.put((index, img))
            for _ in processers:
                q_tasks.put('done')
            for process in processers:
                process.join()

            mat_out = numpy.zeros((len(imgs), 2))
            while not q_results.empty():
                index, bright, kp = q_results.get()
                mat_out[index, :] = bright, kp

            pickle.dump(mat_out, f_out)


if __name__ == '__main__':
    script_nima()


