# -*- coding: utf-8 -*-
"""
Created on 9/11/18
@author: MRChou

Scenario: Functions for exmaining the content of TFRecord format.
"""

import tensorflow as tf
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import pickle


def _parse_cls(dataset, num_to_show=1):
    iter_dataset = dataset.make_one_shot_iterator()
    item = iter_dataset.get_next()

    count = 0
    for i in range(num_to_show):
        count += 1
        try:
            with tf.Session() as sess:
                result = tf.train.Example.FromString(sess.run(item))
                feature = result.features.feature

                labels = feature['image/class/text'].bytes_list.value
                labels = [label.decode() for label in labels]

                img_bytes = feature['image/encoded'].bytes_list.value[0]
                img_bytes = io.BytesIO(img_bytes)
                img = mpimg.imread(img_bytes, format='jpg')

                for label_count, label in enumerate(labels):
                    cv2.putText(img,
                                _to_description(label),
                                (50, label_count*50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                thickness=2,
                                color=(255, 0, 0))

                plt.title(feature['image/filename'].bytes_list.value[0])
                plt.imshow(img)
                plt.show()

        except tf.errors.OutOfRangeError:
            break

    print('Done: ', count)


def _parse_oid(dataset, num_to_show=5):
    iter_dataset = dataset.make_one_shot_iterator()
    item = iter_dataset.get_next()

    count = 0
    with tf.Session() as sess:
        for i in range(num_to_show):
            count += 1
            try:
                result = tf.train.Example.FromString(sess.run(item))
                feature = result.features.feature

                xmins = feature['image/object/bbox/xmin'].float_list.value
                xmaxs = feature['image/object/bbox/xmax'].float_list.value
                ymins = feature['image/object/bbox/ymin'].float_list.value
                ymaxs = feature['image/object/bbox/ymax'].float_list.value
                labels = feature['image/object/class/index'].int64_list.value
                labels = [label for label in labels]

                img_bytes = feature['image/encoded'].bytes_list.value[0]
                img_bytes = io.BytesIO(img_bytes)
                img = mpimg.imread(img_bytes, format='jpg')

                if any(labels):
                    xmins = [int(value) if value > 1 else int(value * img.shape[1]) for value in xmins]
                    xmaxs = [int(value) if value > 1 else int(value * img.shape[1]) for value in xmaxs]
                    ymins = [int(value) if value > 1 else int(value * img.shape[0]) for value in ymins]
                    ymaxs = [int(value) if value > 1 else int(value * img.shape[0]) for value in ymaxs]

                    for xmin, xmax, ymin, ymax, label in zip(xmins, xmaxs,
                                                             ymins, ymaxs,
                                                             labels):
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                      (255, 0, 0))
                        cv2.putText(img, _to_description(label), (xmin, ymax),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, thickness=2, color=(255, 0, 0))

                print(img.shape)
                print('labels: {}, [{},{},{},{}]'.format(labels, xmins, xmaxs, ymins, ymaxs))
                plt.title(feature['image/filename'].bytes_list.value[0])
                plt.imshow(img)
                plt.show()

            except tf.errors.OutOfRangeError:
                break

    print('Done: ', count)


def read_tfr(filename):
    return tf.data.TFRecordDataset(filename)


# "/archive/OpenImg/train_TFRs/train_0145-1000.tfrecord"
# "/archive/Inclusive/train_TFRs/train_0001.tfrecord"
# "/archive/RSNA/train_TFRs/train_0001.tfrecord"
TESTFILE = "/archive/RSNA/train_TFRs/train_0001.tfrecord"

# '/archive/OpenImg/LabelName_to_Description.pkl'
# '/archive/Inclusive/LABELS_TO_DESCRIPTION.pkl'
# NAME_TO_DES = {'1': '1', '0': '0'}
with open('/archive/Inclusive/LABELS_TO_DESCRIPTION.pkl', 'rb') as fin:
    NAME_TO_DES = pickle.load(fin)


def _to_description(labelname):
    return NAME_TO_DES.get(labelname, 'Unknown')


if __name__ == '__main__':
    dataset = read_tfr(TESTFILE)
    _parse_oid(dataset, num_to_show=10)
    # _parse_cls(dataset)

