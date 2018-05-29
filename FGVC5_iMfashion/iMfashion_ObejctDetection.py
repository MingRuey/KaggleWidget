# -*- coding: utf-8 -*-
"""
Created on Fri May 15 07:00:00 2018
@author: MRChou

A attempt on getting cloth location from iMfashion images:
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

Most of the code are just modified from object_detection_tutorial.ipynb in:
https://github.com/tensorflow/models/tree/master/research/object_detection
"""

import os
import cv2
import numpy
import tensorflow as tf
import queue
import threading

# for loading the label map
from object_detection.utils import label_map_util

# for downloading the graph
import six.moves.urllib as urllib
import tarfile
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


def download_graph(name):
    """Download the graph with given name from DOWNLOAD_BASE"""
    model_name = name + '.tar.gz'

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + model_name, model_name)
    tar_file = tarfile.open(model_name)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


def load_detectiongraph(path):
    """Load the graph with given path"""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def load_label(path, num_of_classes):
    label_map = label_map_util.load_labelmap(path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_of_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


# Mostly copy-paste from the source mentioned above.
# Using multi-threading to process the detection.
STORE = '/archive/iMfashion/preprocess/imgs_test/'


def get_inferences(path, graph, que, num_of_threads):
    """Put the results of object detection into que by multi-thred workers"""

    def img_loader():
        """load image from the *path"""
        for file in os.listdir(path):
            if file.lower().endswith('.jpg'):
                if not os.path.isfile(os.path.join(STORE, file)):  # if the file already exists, pass it.
                    yield (file, cv2.imread(os.path.join(path, file))[..., ::-1])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    with graph.as_default():
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        tensor_dict = {}
        for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        imgs = img_loader()

        def detect_sess(threshold=0.5):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                for file, img in imgs:
                    detect = sess.run(tensor_dict, feed_dict={image_tensor: numpy.expand_dims(img, 0)})

                    # all outputs are float32 numpy arrays, so convert types to numpy to accelarate
                    detect['detection_classes'] = detect['detection_classes'][0]
                    detect['detection_boxes'] = detect['detection_boxes'][0]
                    detect['detection_scores'] = detect['detection_scores'][0]

                    boxes = detect['detection_boxes'][numpy.where(
                                    (detect['detection_classes'] == 1) & (detect['detection_scores'] > threshold)
                                    )]
                    boxes = boxes if boxes.size else numpy.array([[0, 0, 1, 1]])

                    que.put((file, img, boxes))

        detect_threads = []
        for _ in range(num_of_threads):
            detect_threads.append(threading.Thread(target=detect_sess()))
        for t in detect_threads:
            t.start()
        for t in detect_threads:
            t.join()


class GrabAndSave(threading.Thread):
    """GrabCut and store the image with grabcut rectangle are the object-detection boxes by get_inferences()."""
    def __init__(self, que, store_path=STORE):
        super(GrabAndSave, self).__init__()
        self._que = que
        self.path = store_path

    def run(self):
        """Get the img and boxes from the que, and do the Grabcut"""
        while True:
            task = self._que.get()
            if task == 'done':
                break
            file, img, boxes = task

            # grab and save image
            img_grab = numpy.zeros(img.shape)
            for box in boxes:
                # scale box to image dimension
                box = numpy.round(box * numpy.array([img.shape[0], img.shape[1], img.shape[0]-1, img.shape[1]-1]))
                ymin, xmin, ymax, xmax = box.astype('uint')

                # get the grabbed content from img, assign it to grab
                mask = numpy.zeros(img.shape[:2], dtype=numpy.uint8)
                cv2.grabCut(img,
                            mask=mask,
                            rect=(xmin, ymin, xmax-xmin, ymax-ymin),
                            bgdModel=None,
                            fgdModel=None,
                            iterCount=5,
                            mode=cv2.GC_INIT_WITH_RECT)
                mask = numpy.where((mask == 3) | (mask == 1), 1, 0)
                img_grab += img * mask[:, :, numpy.newaxis]

            cv2.imwrite(os.path.join(self.path, file), img_grab[..., ::-1])


def build_grab_processer(que, num_of_workers):
    """Create many GrabAndSave instance as multi-thread workers"""
    workers = []
    for _ in range(num_of_workers):
        worker = GrabAndSave(que)
        worker.start()
        workers.append(worker)
    return workers


def main():
    imgs_path = '/rawdata/FGVC5_iMfashion/imgs_test/'
    graph = '/archive/iMfashion/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

    # queue for data exchange between detection and grab
    q = queue.Queue()

    # Multi-thread grabbing
    processers = build_grab_processer(q, 50)

    # Make the detection generator ready
    detection_graph = load_detectiongraph(graph)
    get_inferences(imgs_path, detection_graph, q, 50)

    for _ in processers:
        q.put('done')
    for process in processers:
        process.join()


if __name__ == '__main__':
    main()
