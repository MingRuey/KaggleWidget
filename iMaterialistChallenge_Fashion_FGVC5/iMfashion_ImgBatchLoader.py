"""
Created on Fri Apr 28 07:00:00 2018
@author: MRChou

Read images in the given dir, and output a batch data via self.batch as a generator.
A tool for used by the VGG16 model, for iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""

import numpy
import os.path
import imageio
import pickle
from random import shuffle as rndshuffle
from skimage.transform import resize


class ImgBatchLoader():

    def __init__(self, img_path, img_label):
        self._path = img_path
        self._imglabels = pickle.load(open(os.path.join(self._path, img_label), 'rb'))


    # Generator for loading and resizing images:
    def __load_img(self, shuffle=True):
        files = os.listdir(self._path)
        if shuffle:
            rndshuffle(files)

        for file in files:
            if file.endswith('.jpg'):
                imgid = int(file[:-4])
                try:
                    img = imageio.imread(os.path.join(self._path, file))
                    # currently it forces resizing image to shape (300,300,3)
                    img = resize(img, (300, 300, 3), preserve_range = True)
                    label = self._imglabels[imgid-1, 1:]
                    yield img, label
                except IOError as err:
                    print('While loading {0}: {1}'.format(file, err))


    # A batch-data generator, loops infinitely
    def generator(self, batch_size, avgbatch=False, shuffle=True):
        epoch=1
        while True:
            print('epoch: ', epoch)
            epoch+=1
            # begining of an epoch: loading imgs
            self._imgs = self.__load_img(shuffle=shuffle)

            # start collecting one batch
            count = 0
            img_batch = numpy.zeros((batch_size, 300, 300, 3))
            label_batch = numpy.zeros((batch_size, 1, self._imglabels.shape[1]-1))
            for img, imglabel in self._imgs:
                img_batch[count, :, :, :] = img
                label_batch[count, :, :] = imglabel
                count +=1
                if count == batch_size:
                    count = 0
                    if avgbatch: # subtract mean if necessary
                        mean = numpy.mean(batch, axis=(0,1,2))
                        batch[:, :, :, 0] -= mean[0]
                        batch[:, :, :, 1] -= mean[1]
                        batch[:, :, :, 1] -= mean[2]
                    yield img_batch, label_batch

# Test the module
def main():
    path = '/home/mrchou/code/KaggleWidget/'

    def test():
        # create test images
        for i in range(10):
            imageio.imwrite(path + str(i) + '.jpg', 10 * (i + 1) * numpy.ones((300, 300, 3), dtype='uint8'))

        # create pickle file
        label = numpy.zeros((10, 15))
        for i in range(10):
            label[i, 0] = i
            label[i, i + 1] = 1

        fw = open(path + 'labels.pickle', 'wb')
        pickle.dump(label, fw)
        fw.close()

    test()
    s = ImgBatchLoader(img_path=path, img_label='labels.pickle')
    for i in s.generator(2, shuffle=True):
        print('label 0: ', i[1][0])
        print('img_0', i[0][0, :, :])
        print('label 1: ', i[1][1])
        print('img_1', i[0][1, :, :])

if __name__=='__main__':
    main()