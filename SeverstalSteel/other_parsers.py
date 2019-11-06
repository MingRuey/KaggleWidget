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
            img = tf.image.per_image_standardization(img)

            return img, img_name, label

        return parse_tfexample


class SEG_MASK_FMT(DataFormat):
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
    def get_parser(aug=False):
        """Construct image classification tfexample parser

        Args:
            n_class: the number of class for one-hoted label

        Returns:
            A parser function which takes tfexample as input and return tensors
        """

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=SEG_MASK_FMT.features
            )

            img_name = parsed_feature['filename']
            label = tf.io.decode_raw(parsed_feature['class'], tf.uint8)
            label = tf.cast(label, tf.float32)
            label = tf.cond(
                tf.size(label) > 0,
                true_fn=lambda: tf.cast(tf.reshape(label, IMG_SHAPE), dtype=tf.int32),
                false_fn=lambda: tf.zeros(shape=IMG_SHAPE, dtype=tf.int32)
            )
            label = tf.one_hot(label, depth=5, axis=-1, dtype=tf.int32)
            label = label[..., 1:]

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(parsed_feature['encoded'], channels=3)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
            img = tf.image.per_image_standardization(img)

            if aug:
                # random horizontal flip
                hori_flip = tf.random.uniform([]) > 0.5
                img = tf.cond(hori_flip, lambda: tf.image.flip_left_right(img), lambda: img)
                label = tf.cond(hori_flip, lambda: tf.image.flip_left_right(label), lambda: label)

                # random vertical flip
                vert_flip = tf.random.uniform([]) > 0.5
                img = tf.cond(vert_flip, lambda: tf.image.flip_up_down(img), lambda: img)
                label = tf.cond(vert_flip, lambda: tf.image.flip_up_down(label), lambda: label)

            return img, img_name, label

        return parse_tfexample


class CLSFMT(DataFormat):

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
    def get_parser(aug=False):
        """Construct image classification tfexample parser

        Args:
            n_class: the number of class for one-hoted label

        Returns:
            A parser function which takes tfexample as input and return tensors
        """

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=CLSFMT.features
            )

            label = tf.io.decode_raw(parsed_feature['class'], tf.uint8)
            label = tf.cast(label, tf.float32)
            label = tf.cond(
                tf.size(label) > 0,
                true_fn=lambda: tf.constant([1], dtype=tf.int32),
                false_fn=lambda: tf.constant([0], dtype=tf.int32)
            )

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(parsed_feature['encoded'], channels=3)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
            img = tf.image.per_image_standardization(img)

            if aug:
                # random horizontal flip
                hori_flip = tf.random.uniform([]) > 0.5
                img = tf.cond(hori_flip, lambda: tf.image.flip_left_right(img), lambda: img)

                # random vertical flip
                vert_flip = tf.random.uniform([]) > 0.5
                img = tf.cond(vert_flip, lambda: tf.image.flip_up_down(img), lambda: img)

            return img, label

        return parse_tfexample


class CropClsFMT(DataFormat):

    DIVIDE_Y = 1
    DIVIDE_X = 4

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
    def get_parser(aug=False):
        """Construct image classification tfexample parser

        Args:
            n_class: the number of class for one-hoted label

        Returns:
            A parser function which takes tfexample as input and return tensors
        """

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=CropClsFMT.features
            )

            if CropClsFMT.DIVIDE_Y != 1:
                y_step = IMG_SHAPE[0] // CropClsFMT.DIVIDE_Y
                randy = tf.random.uniform(shape=(), minval=0, maxval=IMG_SHAPE[0] - y_step, dtype=tf.int32)
            else:
                y_step = IMG_SHAPE[0]
                randy = 0

            if CropClsFMT.DIVIDE_X != 1:
                x_step = IMG_SHAPE[1] // CropClsFMT.DIVIDE_X
                randx = tf.random.uniform(shape=(), minval=0, maxval=IMG_SHAPE[1] - x_step, dtype=tf.int32)
            else:
                x_step = IMG_SHAPE[1]
                randx = 0

            label = tf.io.decode_raw(parsed_feature['class'], tf.uint8)
            label = tf.cast(label, tf.float32)
            label = tf.cond(
                tf.size(label) > 0,
                true_fn=lambda: tf.cast(tf.reshape(label, IMG_SHAPE), dtype=tf.int32),
                false_fn=lambda: tf.zeros(shape=IMG_SHAPE, dtype=tf.int32)
            )
            label = label[randy:randy+y_step, randx:randx+x_step]
            label = tf.math.minimum(tf.reduce_sum(label), 1)

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(parsed_feature['encoded'], channels=3)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
            img = tf.image.per_image_standardization(img)
            img = img[randy:randy+y_step, randx:randx+x_step, :]

            if aug:
                # random horizontal flip
                hori_flip = tf.random.uniform([]) > 0.5
                img = tf.cond(hori_flip, lambda: tf.image.flip_left_right(img), lambda: img)

                # random vertical flip
                vert_flip = tf.random.uniform([]) > 0.5
                img = tf.cond(vert_flip, lambda: tf.image.flip_up_down(img), lambda: img)

            return img, label

        return parse_tfexample


class CropInferenceFMT(DataFormat):

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
    def get_parser():
        """Construct image classification tfexample parser

        Args:
            n_class: the number of class for one-hoted label

        Returns:
            A parser function which takes tfexample as input and return tensors
        """

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=CropClsFMT.features
            )

            label = tf.io.decode_raw(parsed_feature['class'], tf.uint8)
            label = tf.cast(label, tf.float32)
            label = tf.cond(
                tf.size(label) > 0,
                true_fn=lambda: tf.constant([1], dtype=tf.int32),
                false_fn=lambda: tf.constant([0], dtype=tf.int32)
            )

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(parsed_feature['encoded'], channels=3)
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)
            img = tf.image.per_image_standardization(img)

            y_step = IMG_SHAPE[0] // CropClsFMT.DIVIDE_Y
            x_step = IMG_SHAPE[1] // CropClsFMT.DIVIDE_X
            img_stack = []
            for y in range(CropClsFMT.DIVIDE_Y):
                img_crop = img[y*y_step:(y+1)*y_step, ...]
                for x in range(CropClsFMT.DIVIDE_X):
                    img_stack.append(img_crop[:, x*x_step:(x+1)*x_step, ...])

            img = tf.stack(img_stack, axis=0)
            return img, label

        return parse_tfexample


if __name__ == "__main__":
    pass
