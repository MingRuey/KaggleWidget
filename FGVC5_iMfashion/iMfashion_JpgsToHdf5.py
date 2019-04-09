# -*- coding: utf-8 -*-
"""
Created on May. 05. 2018
@author: MRChou

Load all jpgs in side a directory, resize them with specified shape and store into a Hdf5 file,

"""

import os
import h5py
import imageio
from skimage.transform import resize


def JpgsToHdf5(img_path, hdf5_name, img_size=(300,300,3), dtype='uint8'):

    hdf5 = h5py.File(hdf5_name, mode='w')
    imgs = [file for file in os.listdir(img_path) if file.lower().endswith('jpg')]
    imgs.sort(key=lambda x: int(x[:-4])) # sort images according to image id

    # create Group for images
    hdf5.create_dataset('imgs', (len(imgs), *img_size), dtype=dtype)

    # create Group for image ids
    hdf5.create_dataset('ids', (len(imgs), ), dtype='uint32')

    k = 0
    for jpg in imgs:
        id = int(jpg[:-4])
        img = imageio.imread(os.path.join(img_path, jpg))
        img = resize(img, img_size, mode='edge', preserve_range=True).astype(dtype)

        hdf5['imgs'][ k ,:,:,:] = img
        hdf5['ids'][ k ] = id
        k +=1

    hdf5.close()

def main():
    img_path  = '/rawdata/FGVC5_iMfashion/imgs_train/'
    hdf5_name = '/archive/iMfashion/imgs_hd5f/train_300x300x3.hdf5'

    JpgsToHdf5(img_path=img_path, hdf5_name=hdf5_name, img_size=(300,300,3), dtype='uint8')

if __name__=='__main__':
    main()