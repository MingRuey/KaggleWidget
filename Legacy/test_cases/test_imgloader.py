"""

Created on May 31 20:00 2018
@author: MRChou

The test cased for Keras_ImgLabelLoader and iMfashion_ImgBatchLoader

"""

import os
import pickle
import numpy
import cv2
from Legacy.KerasUtils import ImgLabelLoader
from FGVC5_iMfashion.iMfashion_ImgBatchLoader import ImgBatchLoader


def test_ImgLabelLoader():
    path = '/home/mrchou/code/KaggleWidget/'

    def test():
        # create test images and pickle file
        label = numpy.zeros((10, 15))

        for i in range(10):
            cv2.imwrite(path + str(i) + '.jpg', 10 * (i + 1) * numpy.ones((300, 300, 3), dtype='uint8'))
            label[i, 0] = i
            label[i, i + 1] = 1

        fw = open(path + 'labels.pickle', 'wb')
        pickle.dump(label, fw)
        fw.close()

    test()

    imgs = [i for i in os.listdir(path) if i.endswith('jpg')]
    print(imgs.sort())
    s = ImgLabelLoader(imgs=imgs, labels=pickle.load(open(path+'labels.pickle', 'rb')))
    for i in s.batch_gener(2, epoch=2, shuffle=True):
        print('label 0: ', i[1][0])
        print('img_0', i[0][0, 0, 0])
        print('label 1: ', i[1][1])
        print('img_1', i[0][1, 0, 0])


def test_iMfashion_ImgBatchLoader():
    path = '/home/mrchou/code/KaggleWidget/'

    def test():
        # create test images and pickle file
        label = numpy.zeros((10, 15))

        for i in range(10):
            cv2.imwrite(path + str(i) + '.jpg', 10 * (i + 1) * numpy.ones((300, 300, 3), dtype='uint8'))
            label[i, 0] = i
            label[i, i + 1] = 1

        fw = open(path + 'labels.pickle', 'wb')
        pickle.dump(label, fw)
        fw.close()

    test()
    s = ImgBatchLoader(img_path=path, img_label='labels.pickle')
    for i in s.generator(2, epoch=2, shuffle=True):
        print('label 0: ', i[1][0])
        print('img_0', i[0][0, 0, 0])
        print('label 1: ', i[1][1])
        print('img_1', i[0][1, 0, 0])


if __name__ == '__main__':
    pass
