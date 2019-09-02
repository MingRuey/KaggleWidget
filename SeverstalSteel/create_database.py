import os
import sys
import pathlib
import logging
import math
from collections import defaultdict
from typing import NamedTuple
from shutil import copyfile

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

file = os.path.basename(__file__)
file = pathlib.Path(file).stem
file = pathlib.Path(os.path.dirname(__file__)).joinpath(file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(message).1000s ',
    handlers=[
        logging.FileHandler("{}.log".format(file)),
        logging.StreamHandler(sys.stdout)
        ]
    )

import numpy as np
import tensorflow as tf  # noqa: E402
from MLBOX.Database.formats import _tffeature_bytes, _tffeature_float, _tffeature_int64   # noqa: E402
from MLBOX.Database.formats import DataFormat, IMGFORMAT  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402
from variables import train_csv  # noqa: E402


IMG_SHAPE = (256, 1600)


def _convert_index_to_pixel_loc(index: int, img_shape: tuple) -> tuple:
    """Given an index of pixel, find its x, y position in image

    Args:
        index:
            the index of pixel,
            which is counted from top to bottom then from left to right
            also, it's ONE-INDEXD (i.e. 1 is the first pixel)
        img_shape:
            a tuple of int, image shape in (height, width)

    Return:
        a tuple of int, the location of pixel in (y, x).
        the y,x is ZERO-INDEXED (i.e. (0, 0) is the first pixel)
    """
    msg = "label outside range of img(h={}, w={}), got index: {}"
    h, w = img_shape
    assert math.ceil(index / h) <= w, msg.format(h, w, index)
    return (index % h or h) - 1, (index-1) // h


def create_array_from_labels(labels: dict, img_shape: tuple=IMG_SHAPE) -> np.array:
    """Create a np.array which contains the label information

    Args:
        labels:
            A dictionary map class index to labels

    Return:
        np.array contains the label information,
        where the pixel with labeled are marked with class_idx
    """
    label_img = np.zeros(shape=img_shape, dtype="int8")
    for cls_idx, cls_labels in labels.items():
        for idx, cnt in zip(cls_labels[::2], cls_labels[1::2]):
            h1, w1 = _convert_index_to_pixel_loc(idx, img_shape)
            h2, w2 = _convert_index_to_pixel_loc(idx + cnt - 1, img_shape)
            label_img[h1:h2+1, w1:w2+1] = cls_idx

    assert not label_img.nonzero()[0].shape[0], str(labels)
    return label_img


def get_label_map(csv_file):
    file = pathlib.Path(csv_file)
    if not file.is_file():
        raise ValueError("Invalid csv file: {}".format(csv_file))

    id_map = {}
    with open(str(file), "r") as f:
        next(f)
        for line in f:
            imgid, labels = line.split(",")
            imgid, class_idx = imgid.split(".")
            assert len(imgid) == 9, imgid

            class_idx = int(class_idx[-1])
            assert 1 <= class_idx <= 4

            labels = labels.split()
            if labels:
                cls_label = [int(label) for label in labels]
                assert len(labels) % 2 == 0

                id_map.setdefault(imgid, {class_idx: labels})[class_idx] = \
                    cls_label

    return id_map


class SEGFORMAT(DataFormat):
    """Format for classification on image"""

    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'extension': tf.io.FixedLenFeature([], tf.string),
        'encoded': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.FixedLenFeature([], tf.string, default_value=b"")
    }
    valid_extensions = {'.jpg', '.bmp'}

    def __init__(self, img_label_map=None):
        """
        Args:
            img_label_map:
                a dictionary that maps img_id into int or a list of int.
                which specify the label(s) of the image.
                If set to None,
                no class label is included in the tf.train.Example
        """
        self._label_map = None if not img_label_map else img_label_map.copy()

    @staticmethod
    def load_from_file(file: str):
        """Load image from file and return image id and content in bytes

        Args:
            img_file: a string specify file path of image

        Return:
            a tuple of bytes (image id, file extension, content in bytes)

        Raise:
            OSError: when file not exist
            TypeError: when file exist but not recognized as image
        """
        path = pathlib.Path(file)
        if not path.is_file():
            raise OSError("Invalid file path")

        image_id = path.stem
        image_type = imghdr.what(str(path))
        if not image_type:
            raise TypeError("Unrecognized image type")

        with open(str(file), 'rb') as f:
            image_bytes = f.read()

        return (
            bytes(image_id, 'utf8'),
            bytes(image_type, 'utf8'),
            image_bytes,
            )

    def to_tfexample(self, img_file_path):
        """Load image and return a tf.train.Example object

        Args:
            img_file_path:
                a string specify the path of image
        """
        img_id, img_type, img_bytes = IMGFORMAT.load_from_file(img_file_path)

        fields = {
            'filename': _tffeature_bytes(img_id),
            'extension': _tffeature_bytes(img_type),
            'encoded': _tffeature_bytes(img_bytes),
            }

        if self._label_map:
            labels = self._label_map.get(img_id.decode('utf8'))
            if labels:
                label_array = create_array_from_labels(labels=labels)
                fields['class'] = _tffeature_bytes(label_array.tostring())

        return tf.train.Example(features=tf.train.Features(feature=fields))

    @staticmethod
    def get_parser(n_class: int):
        """Construct image classification tfexample parser

        Args:
            n_class: the number of class for one-hoted label

        Returns:
            A parser function which takes tfexample as input and return tensors
        """

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=SEGFORMAT.features
            )

            img_name = parsed_feature['filename']
            label = tf.io.decode_raw(parsed_feature['class'], tf.uint8)
            label = tf.cast(label, tf.float32)
            label = tf.cond(
                tf.size(label) > 0,
                true_fn=lambda: tf.cast(tf.reshape(label, IMG_SHAPE), dtype=tf.int32),
                false_fn=lambda: tf.zeros(shape=IMG_SHAPE, dtype=tf.int32)
            )
            label = tf.one_hot(label, depth=n_class, axis=-1, dtype=tf.int32)

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(parsed_feature['encoded'], channels=3)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)

            return img, img_name, label

        return parse_tfexample


if __name__ == "__main__":

    train_labels = get_label_map(train_csv)

    print("train data count: ", len(train_labels))

    dataformat = SEGFORMAT(img_label_map=train_labels)
    dataset = DataBase(formats=dataformat)
    dataset.build_database(
        input_dir="/rawdata/Severstal_Steel/imgs_train",
        output_dir="/archive/Steel/database/train/"
    )

    dataset = DataBase(formats=SEGFORMAT())
    dataset.build_database(
        input_dir="/rawdata/Severstal_Steel/imgs_test",
        output_dir="/archive/Steel/database/test/"
    )
