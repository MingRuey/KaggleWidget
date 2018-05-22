# -*- coding: utf-8 -*-
"""
Created on May. 05. 2018
@author: MRChou

Filt the images in train set, so that whose labels are closest to validation set are left.
"Closest" is defined by the inner product of labels,
the label of an image in train set is inner-doted to all images in validation set,
then sum  all inner product values to be the distance for a image to validation set.

"""

import os
from shutil import copyfile
import pickle
import cupy
import numpy


def one_fold_dot(train_label, vali_label, device=0):
    with cupy.cuda.Device(device):
        train_label = cupy.array(train_label, dtype='int8')
        vali_label = cupy.array(vali_label, dtype='int8')

        # inner dot train_label and vali_label
        dis_mat = cupy.dot(train_label, cupy.transpose(vali_label))

        # Each row is normalize by length of train label.
        dis_mat = dis_mat / cupy.linalg.norm(train_label, axis=1)[:, None]
        dis_mat = dis_mat.sum(axis=1)
    return cupy.asnumpy(dis_mat)


def closest_imgs(train_label, vali_label, num_of_img, k_fold=4):
    """ compute the distance matrix D,
        with D_ij = < |train_i|, validation_j>,
        where train_i, vali_j is the label of i-th, j-th image in train, vali,
        || means the vector is normalized."""

    n = train_label.shape[0]
    dis = numpy.zeros(n)

    fold_size = int(numpy.ceil(n / k_fold))
    for i in range(k_fold):
        step = [i*fold_size, (i+1)*fold_size] if i+1 != k_fold else [i*fold_size, n]
        dis[step[0]:step[1]] = one_fold_dot(train_label=train_label[step[0]:step[1], :],
                                            vali_label=vali_label,
                                            device=0
                                            )
    return dis.argsort()[-num_of_img:]


def copy_imgs(imgs_path, ids, out_path):
    for i in ids:
        img_name = str(i)+'.jpg'
        try:
            copyfile(os.path.join(imgs_path, img_name), os.path.join(out_path, img_name))
        except FileNotFoundError:
            print('{0} not found while copying!'.format(img_name))
    return None


def main():
    train_label = '/archive/iMfashion/labels/labels_train.pickle'
    vali_label = '/archive/iMfashion/labels/labels_validation.pickle'

    num_of_img = 200000

    train_label = pickle.load(open(train_label, 'rb'))
    vali_label = pickle.load(open(vali_label, 'rb'))

    ids = closest_imgs(train_label[:, 1:], vali_label[:, 1:],
                       num_of_img=num_of_img,
                       k_fold=8)
    ids = train_label[:, 0][ids]

    train_path = '/rawdata/FGVC5_iMfashion/imgs_train/'
    out_path = '/archive/iMfashion/imgs_train_split/kNN_train/'

    copy_imgs(train_path, ids, out_path)


if __name__ == '__main__':
    main()
