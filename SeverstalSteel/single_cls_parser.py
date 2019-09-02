import os
import sys
import pathlib
import logging

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

import tensorflow as tf  # noqa: E402
from MLBOX.Database.formats import _tffeature_bytes, _tffeature_float, _tffeature_int64   # noqa: E402
from MLBOX.Database.formats import DataFormat, IMGFORMAT  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402


IMG_SHAPE = (256, 1600)


class SEG_SINGLECLS_FMT(DataFormat):
    """Format for parse only single class label"""

    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'extension': tf.io.FixedLenFeature([], tf.string),
        'encoded': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.FixedLenFeature([], tf.string, default_value=b"")
    }

    @staticmethod
    def load_from_file(file: str):
        raise NotImplementedError()

    def to_tfexample(self, img_file_path):
        raise NotImplementedError()

    @staticmethod
    def get_parser(target_class: int):
        """Construct image classification tfexample parser

        Args:
            n_class: the number of class for one-hoted label

        Returns:
            A parser function which takes tfexample as input and return tensors
        """

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=SEG_SINGLECLS_FMT.features
            )

            img_name = parsed_feature['filename']
            label = tf.io.decode_raw(parsed_feature['class'], tf.uint8)
            label = tf.cast(label, tf.float32)
            label = tf.cond(
                tf.size(label) > 0,
                true_fn=lambda: tf.cast(tf.reshape(label, IMG_SHAPE), dtype=tf.int32),
                false_fn=lambda: tf.zeros(shape=IMG_SHAPE, dtype=tf.int32)
            )
            label = tf.one_hot(tf.cast(tf.equal(label, target_class), tf.int32), depth=2)

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(parsed_feature['encoded'], channels=3)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)

            return img, img_name, label

        return parse_tfexample


if __name__ == "__main__":
    pass
