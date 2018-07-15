# -*- coding: utf-8 -*-
"""
Created on July 11 19:59 2018
@author: MRChou

Convert training imgs into TFRecord file format.
"""

import os
import time
import pickle
import threading
import queue

import cv2
import tensorflow as tf

from CnnUtils import ObjDetectImg

with open('/archive/OpenImg/ImgId_to_BboxLabels.pkl', 'rb') as f:
    # defaultdict mapping image id to labels.
    # return a list of BBox label or [] if not found.
    IMG_TO_LABELS = pickle.load(f)


def to_imgobj(file):
    """Read image file and get labels, return an Image class."""
    try:
        img = cv2.imread(file)[::-1]  # turn image into RGB from BGR.
        if img is None:
            raise OSError('File not exist %s' % file)
    except (OSError, TypeError) as err:
        print('Error when loading %s, %s' % (os.path.basename(file), err))
    else:
        imgid = os.path.splitext(os.path.basename(file))[0]
        labels = IMG_TO_LABELS[imgid]
        if labels:
            return ObjDetectImg(imgid, img, labels)
        else:
            print('No labels found with %s' % (os.path.basename(file)))
            return None


def to_tfexample(features):
    """Return an tf.train.Example from an dictionary of features"""
    return tf.train.Example(
        features=tf.train.Features(feature=features)
    )


class ImgWorker(threading.Thread):
    """An thread for reading image and turn it into tf.train.Example"""

    def __init__(self, img_files, que):
        super(ImgWorker, self).__init__()
        self.files = img_files
        self.que = que

    def run(self):
        for file in self.files:
            img = to_imgobj(file)
            if img is not None:
                tfexample = to_tfexample(img.tffeatures())
                self.que.put(tfexample)


def img_to_tfrecord(img_files, fout, num_of_workers):
    """Read images and write tfexample into a tfrecord file,
    accelerating by multiprocess"""

    img_files = (file for file in img_files)

    que = queue.Queue()
    workers = [ImgWorker(img_files, que) for _ in range(num_of_workers)]
    for worker in workers:
        worker.start()

    writer = tf.python_io.TFRecordWriter(fout)
    while True:
        try:
            tf_example = que.get(timeout=20)
            writer.write(tf_example.SerializeToString())
        except queue.Empty:
            print('Finish writing!')
            writer.close()
            break

    for worker in workers:
        worker.join()


if __name__ == '__main__':
    img_path = '/rawdata/Google_OpenImg/imgs_train/'
    outfile = '/archive/OpenImg/train.tfrecord'

    start_t = time.time()
    files = [img_path + file
             for file in os.listdir(img_path)[:5000] if file.endswith('.jpg')]
    img_to_tfrecord(files, outfile, num_of_workers=16)

    print('Task finished in %s seconds' % (time.time() - start_t))
