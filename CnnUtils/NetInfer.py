# -*- coding: utf-8 -*-
"""
Created on July 19 02:45 2018
@author: MRChou

Functions help to make evaluation/prediction from CNN on images.

"""

import queue
import threading
from collections import namedtuple
from functools import partial

import numpy
import tensorflow as tf

_bbox = namedtuple('InferenceBbox', ['LabelID',
                                     'Confidence',
                                     'XMin',
                                     'XMax',
                                     'YMin',
                                     'YMax'])
_num_of_cores = 16


def _tf_get_iter_from_files(files, map_func):  # TODO: batchsize now is simply 1
    """A tf Op creates a tf.data.Dataset.make_one_shot_iterator() from files,
    after mapping them with map_func"""
    files = tf.constant(files, name='files')
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(map_func,
                          num_parallel_calls=_num_of_cores
                          ).prefetch(10)
    return dataset.make_one_shot_iterator()


def _tf_parse_img(filename, img_size):
    """A tf Op reads image and resizes it """
    img = tf.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img,
                                 list(img_size),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def _detect_to_predict(detections):  # TODO: for OID API pretrained &batchsize=1
    """Get a list of bbox objs from model prediction"""
    labels = detections[0][0]
    scores = detections[1][0]
    boxes = detections[2][0]

    target_index = numpy.where(scores > 0)

    labels = labels[target_index]
    scores = scores[target_index]
    boxes = boxes[target_index]

    return labels, scores, boxes


class Model:

    def __init__(self):
        self.modelgraph_def = None

    def load_model(self, model_file_path):
        """Load graph from file, store it as tf.Graph in self.modelgraph"""
        with tf.gfile.GFile(model_file_path, 'rb') as fid:
            modelgraph_def = tf.GraphDef()
            modelgraph_def.ParseFromString(fid.read())

        self.modelgraph_def = modelgraph_def

    @staticmethod
    def _get_input_tensor_name():  # TODO: only for OID API pretrained
        """Return the name of input tensor from model graph definition"""
        return 'image_tensor:0'

    @staticmethod
    def _get_output_tensor_name():  # TODO: only for OID API pretrained
        """Return the name of output tensor from model graph definition"""
        return ['detection_classes:0',
                'detection_scores:0',
                'detection_boxes:0']

    def _detect(self, tensor_input):
        """A tf Op that takes tensor_input as input,
        returns output tensor from model graph definition"""
        return tf.import_graph_def(
            self.modelgraph_def,
            input_map={self._get_input_tensor_name(): tensor_input},
            return_elements=self._get_output_tensor_name()
        )

    # TODO: there should be a better way than an external queue(so tired...)
    def infer_on_imgs(self, img_files, que, img_size=(300, 300)):
        """Make inference on imgfiles from model graph def,
        put the result into a queue"""
        if self.modelgraph_def is None:
            raise AttributeError('Model graph def not loaded.')
        else:
            with tf.Graph().as_default():

                # configure input data
                parse_img = partial(_tf_parse_img, img_size=img_size)
                imgs = _tf_get_iter_from_files(img_files, parse_img)
                img = imgs.get_next(name='img')
                img = tf.expand_dims(img, 0)

                # connect input with model._detect
                detections = self._detect(img)

                with tf.Session() as sess:
                    try:
                        while True:
                            que.put(sess.run(detections))
                    except tf.errors.OutOfRangeError:
                        print("Finish Inference.")


if __name__ == '__main__':
    import os
    import time

    imgfiles = ['/archive/OpenImg/eval_TFRs/imgs/' + file for file in os.listdir('/archive/OpenImg/eval_TFRs/imgs')][:500]

    class _PredictWorker(threading.Thread):
        """An thread for reading image and turn it into tf.train.Example"""

        def __init__(self, que):
            super(_PredictWorker, self).__init__()
            self.que = que

        def run(self):
            model = Model()
            model.load_model('/archive/OpenImg/models/FasterRCNN_InceptResV2_Pretrained/frozen_inference_graph.pb')
            model.infer_on_imgs(img_files=imgfiles, que=self.que)


    start_t = time.time()

    q = queue.Queue()
    worker = _PredictWorker(que=q)
    worker.start()

    time.sleep(60)
    with open('/archive/OpenImg/test_eval.csv', 'w') as fout:
        fout.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')

        bbox_line = '{},{:d},{:f},{XMin},{XMax},{YMin},{YMax}\n'
        count = 0
        while True:
            try:
                labels, scores, boxes = _detect_to_predict(q.get(timeout=20))
                imageid = os.path.basename(imgfiles[count].strip('.jpg'))
                for label, score, box in zip(labels, scores, boxes):
                    fout.write(bbox_line.format(
                        imageid, int(label), score,
                        XMin=box[1], XMax=box[3], YMin=box[0], YMax=box[2]
                    ))
                count += 1
            except queue.Empty:
                print('Finish writing!')
                break

    worker.join()

    print('Task finished in %s seconds' % (time.time() - start_t))
