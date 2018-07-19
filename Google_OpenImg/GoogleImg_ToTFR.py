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

import pandas
import tensorflow as tf

from CnnUtils.ImgObj import ObjDetectImg

with open('/archive/OpenImg/ImgId_to_BboxLabels.pkl', 'rb') as f:
    # defaultdict mapping image id to labels.
    # return a list of BBox label or [] if not found.
    IMG_TO_LABELS = pickle.load(f)

with open('/archive/OpenImg/LabelName_to_ClassID.pkl', 'rb') as f:
    # dict mapping from label name to label index
    LABEL_TO_INDEX = pickle.load(f)


# Modified ObjDetectImg to let features include class index
class ObjDetectImgWithIndex(ObjDetectImg):

    def tffeatures(self):
        # get original features
        box_features = super(ObjDetectImgWithIndex, self).tffeatures()

        # find out class indices for each label
        cls_index = []
        for bbox in self.bboxs:
            cls_index.append(LABEL_TO_INDEX[bbox.labelname])  # may get KeyError

        box_features['image/object/class/label'] = \
            tf.train.Feature(int64_list=tf.train.Int64List(value=cls_index))
        return box_features


def to_imgobj(file):
    """Read image file and get labels, return an Image class."""
    with open(file, 'rb') as f:
        img_bytes = f.read()

    imgid = os.path.splitext(os.path.basename(file))[0]
    labels = IMG_TO_LABELS[imgid]
    if not labels:
        raise KeyError('No labels found with %s' % (os.path.basename(file)))
    else:
        # Also get KeyError when creating ObjDetectImgWithIndex
        return ObjDetectImgWithIndex(imgid, img_bytes, labels)


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
            try:
                img = to_imgobj(file)
            except (FileNotFoundError, KeyError) as err:
                print(err)
            else:
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
    PATH = '/rawdata/Google_OpenImg/imgs_train/'

    evalid = '/rawdata/Google_OpenImg/challenge-2018-image-ids-valset-od.csv'
    evalid = {imgid for imgid in pandas.read_csv(evalid).ImageID.values}

    evalfiles = [PATH + imgid + '.jpg' for imgid in evalid]
    trainfiles = [PATH + file for file in os.listdir(PATH) if
                  (not file.strip('.jpg') in evalid) and file.endswith('.jpg')
                  ]

    start_t = time.time()

    # for train images
    outfilename = '/archive/OpenImg/data/train_{:0=4}-{:0=4}.tfrecord'
    num_of_tfrecords = 1000
    len_of_filebatch = -(-len(trainfiles) // num_of_tfrecords)  # == math.ceil

    for i in range(num_of_tfrecords):
        outfile = outfilename.format(i+1, num_of_tfrecords)

        filebatch = trainfiles[i*len_of_filebatch: (i+1)*len_of_filebatch]
        img_to_tfrecord(filebatch, outfile, num_of_workers=16)

    # for eval images
    outfilename = '/archive/OpenImg/data/eval_{:0=4}-{:0=4}.tfrecord'
    num_of_tfrecords = 10
    len_of_filebatch = -(-len(evalfiles) // num_of_tfrecords)  # == math.ceil

    for i in range(num_of_tfrecords):
        outfile = outfilename.format(i + 1, num_of_tfrecords)

        filebatch = evalfiles[i * len_of_filebatch: (i + 1) * len_of_filebatch]
        img_to_tfrecord(filebatch, outfile, num_of_workers=16)

    print('Task finished in %s seconds' % (time.time() - start_t))
