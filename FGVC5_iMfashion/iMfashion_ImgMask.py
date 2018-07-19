# -*- coding: utf-8 -*-
"""
Created on May. 01. 2018.
@author: mrchou

define different masks on image used in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""
import cv2
import numpy
from scipy import fftpack as scifft
from scipy.ndimage import convolve as sciconv
from scipy.ndimage import gaussian_filter as scigauss
import matplotlib.pyplot as plt


# Detection of "important pixels"(=Saliency detection)
# An realization of Saliency Detection: A Spectral Residual Approach by Xiaodi Hou
# checkout: https://ieeexplore.ieee.org/abstract/document/4270292/
def salient(img, viewsize=(5,5)):
    # turn img into grayscale
    z = numpy.mean(img, axis=2)

    # fourier transform, then adjust amplitude domain as the paper suggested.
    z = scifft.fft2(z)
    angle = numpy.angle(z)
    log_amp = numpy.log(numpy.abs(z)+1) # amplitude domain of image, taking abs and +1 to do the log.
    log_amp = log_amp - sciconv(log_amp, numpy.ones(viewsize)/(viewsize[0]*viewsize[1]), mode='constant', cval=0.0)

    # inverse fourier
    z = abs( scifft.ifft2(numpy.exp(log_amp+1j*angle)) ) **2
    z = scigauss(z, viewsize, mode='constant', cval=0.0)
    z = numpy.where(z > z.mean(), 0, 1)
    return z


def canny_split(img):
    """Use canny edge detector to detect images with multiple blocks."""
    canny = cv2.Canny(img, 0, 255)


def main():
    path = '/home/mrchou/code/KaggleWidget/FGVC5_iMfashion/'
    img = cv2.imread(path+'m_single_colorful.jpg')[...,::-1]

    sal = salient(img, viewsize=(20,20))
    sal = sal[:,:, numpy.newaxis]
    #sal = salient(img*sal, viewsize=(5,5))
    #sal = sal[:, :, numpy.newaxis]

    f, axarr = plt.subplots(2)
    axarr[0].imshow(img)
    axarr[1].imshow(img*sal)
    plt.show()


if __name__ == '__main__':
    main()