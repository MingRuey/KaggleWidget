# -*- coding: utf-8 -*-
"""
Created on 9/15/18
@author: MRChou

Scenario: For writing image array and feature labels into TFRecord file.
"""

import time
import logging
from pathlib import PurePath
from threading import Lock
from concurrent import futures
from queue import Queue, Empty

import tensorflow as tf


class ImgObjAbstract:
    """A abstract interface that holds a single image and its labels."""

    def to_tfexample(self):
        """
        Turn image and labels into an tensorflow example.
        Should raise NotImplementedError facing unknown flag.
        """
        raise NotImplementedError


class _ImgGenerKeeper:
    """An helper class record states of generator yielding ImgObj"""

    def __init__(self, imgobj_gener):
        self._imgobj_gener = (imgobj for imgobj in imgobj_gener)
        self._lock = Lock()
        self.success_count = 0
        self.err_count = 0
        self.imggner_terminate = False

    def imgobj_gener(self):
        """Make self_imgobj_gener thread-safe"""
        self._lock.acquire()
        try:
            yield from self._imgobj_gener
        finally:
            self._lock.release()


def _add_num_to_tfrecord_filename(fout, file_count):
    fout = PurePath(fout)
    new_file = fout.stem + '_{:0=4}.tfrecord'.format(file_count)
    return fout.with_name(new_file).as_posix()


def write_tfrecord(imgobj_gener,
                   num_imgs_per_file,
                   fout,
                   num_threads=40):
    """
    Write ImgObj classes into tfrecord file from a generator

    Args:
        imgobj_gener: a generator yield ImgObj classes
        num_imgs_per_file: maximum number of images in sinlge TFRecord file
        fout: the name of output file.
              for example: '/data/head.tfrecord' will results in
              '/data/head-001-158.tfrecord', '.../head-002-158.tfreocrd', ...
              which means there are 158 files in total.
        num_threads: number of threads to read image from file system

    """
    tfexample_queue = Queue()
    imgs = _ImgGenerKeeper(imgobj_gener)

    # a processor turn image to tf.example for multi-thread uses
    def imgobj_processor():
        try:
            for imgobj in imgs.imgobj_gener():
                tfexample_queue.put(imgobj.to_tfexample())
        except Exception as err:
            err_msg = 'Warning: processing imgobjs: {}'
            logging.exception(err_msg.format(err))
            imgs.err_count += 1

        imgs.imggner_terminate = True

    start_time = time.time()
    logging.info('Start writing tfrecords')

    # multi-thread img-to-tfexample processorers
    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        threads = []
        for _ in range(num_threads):
            threads.append(executor.submit(imgobj_processor))

        # a write waiting for tf.example to write
        file_count = 1
        file = _add_num_to_tfrecord_filename(fout, file_count)
        writer = tf.python_io.TFRecordWriter(file)
        while True:
            try:
                tf_example = tfexample_queue.get(timeout=30)
                writer.write(tf_example.SerializeToString())
            except Empty:
                if not imgs.imggner_terminate:
                    msg = 'Exit before image generator terminates.'
                    logging.exception(msg)
                if not all([future.done() for future in threads]):
                    msg = 'Exit before all thread ended.'
                    logging.exception(msg)
                writer.close()
                break
            else:
                imgs.success_count += 1

            if imgs.success_count % num_imgs_per_file == 0:
                msg = 'Finished writing {} to {} image(s) into {}.'
                msg = msg.format((file_count - 1) * num_imgs_per_file,
                                 file_count * num_imgs_per_file,
                                 PurePath(file).name
                                 )
                logging.info(msg)

                file_count += 1
                file = _add_num_to_tfrecord_filename(fout, file_count)
                writer.close()
                writer = tf.python_io.TFRecordWriter(file)

    logging.info(' -- takes %s seconds' % (time.time() - start_time))
    logging.info(' -- %d success/%d errors' % (imgs.success_count, imgs.err_count))
    logging.info(' -- see WriteTFRecord.log')


if __name__ == '__main__':
    pass
