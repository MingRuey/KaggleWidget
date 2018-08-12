# -*- coding: utf-8 -*-
"""
Created on July 19 02:45 2018
@author: MRChou

scripts to make evaluation/prediction from CNN on images.

"""

import os
import time
import queue
import pickle
import threading

from CnnUtils.NetInfer import Model, _detect_to_predict


with open('/archive/OpenImg/LabelName_to_ClassID.pkl', 'rb') as pkl:
    # dict mapping from label name to label index
    LABEL_TO_INDEX = pickle.load(pkl)

with open('/archive/OpenImg/201711_classes.pkl', 'rb') as pkl:
    # old dict mapping from label index to label name
    INDEX_TO_LABEL = pickle.load(pkl)


def _pretrained_to_label(index):
    """Turn Open Image pretrained model index into image label,
    while if the label is not in challenge 500, raise KeyError"""
    if INDEX_TO_LABEL[index] in LABEL_TO_INDEX:
        return INDEX_TO_LABEL[index]
    else:
        raise KeyError('Label only exists in old 2017 file.')


def _write_out_for_eval(fout, input_que):
    """write out csv for evaluation"""
    with open(fout, 'w') as f:
        f.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')

        bbox_line = '{ImageID},{LabelName:d},{Score:f},{XMin},{XMax},{YMin},{YMax}\n'
        count = 0
        while True:
            try:
                filename, labels, scores, boxes = _detect_to_predict(input_que.get(timeout=20))
                imageid = os.path.basename(filename.decode().strip('.jpg'))
                for label, score, box in zip(labels, scores, boxes):
                    try:
                        f.write(bbox_line.format(
                            ImageID=imageid,
                            LabelName=LABEL_TO_INDEX[_pretrained_to_label(int(label))],
                            Score=score,
                            XMin=box[1], XMax=box[3], YMin=box[0], YMax=box[2]
                        ))
                    except KeyError:
                        pass
                count += 1
            except queue.Empty:
                print('Finish writing!')
                break


def _write_out_for_submit(fout, input_que):
    """write out csv for submission file on Kaggle"""
    with open(fout, 'w') as f:
        f.write('ImageID,PredictionString\n')

        bbox_start = '{ImageID},'
        bbox_item = '{LabelName} {Confidence} {XMin} {YMin} {XMax} {YMax} '
        count = 0
        while True:
            try:
                filename, labels, scores, boxes = _detect_to_predict(input_que.get(timeout=20))

                # get image id
                imageid = os.path.basename(filename.decode().strip('.jpg'))
                f.write(bbox_start.format(ImageID=imageid))

                for label, score, box in zip(labels, scores, boxes):
                    try:
                        f.write(bbox_item.format(
                            LabelName=_pretrained_to_label(int(label)),
                            Confidence=score,
                            XMin=box[1],
                            YMin=box[0],
                            XMax=box[3],
                            YMax=box[2]
                        ))
                    except KeyError:
                        pass

                f.write('\n')
                count += 1

            except queue.Empty:
                print('Finish writing!')
                break


def script_for_eval():
    path = '/archive/OpenImg/eval_TFRs/imgs/'
    fout = '/archive/OpenImg/infer_on_eval.csv'
    modelfile = '/archive/OpenImg/models/FasterRCNN_InceptResV2_Pretrained/frozen_inference_graph.pb'

    imgfiles = [path + file for file in os.listdir(path)]

    class _PredictWorker(threading.Thread):
        """An thread for reading image and turn it into tf.train.Example"""

        def __init__(self, que):
            super(_PredictWorker, self).__init__()
            self.que = que

        def run(self):
            model = Model()
            model.load_model(modelfile)
            model.infer_on_imgs(img_files=imgfiles, que=self.que)

    start_t = time.time()

    q = queue.Queue()
    worker = _PredictWorker(que=q)
    worker.start()

    time.sleep(600)
    _write_out_for_eval(fout=fout, input_que=q)
    worker.join()

    print('Finish infer on eval set in %s secs' % (time.time() - start_t))


def script_for_submit():
    path = '/rawdata/Google_OpenImg/imgs_test/'
    fout = 'OI00_submission.csv'
    modelfile = '/archive/OpenImg/models/OI00_FasterRCNN_InceptResV2/export_ckpt34126/frozen_inference_graph.pb'

    imgfiles = [path + file for file in os.listdir(path)]

    class _PredictWorker(threading.Thread):
        """An thread for reading image and turn it into tf.train.Example"""

        def __init__(self, que):
            super(_PredictWorker, self).__init__()
            self.que = que

        def run(self):
            model = Model()
            model.load_model(modelfile)
            model.infer_on_imgs(img_files=imgfiles, que=self.que)

    start_t = time.time()

    q = queue.Queue(maxsize=20)
    worker = _PredictWorker(que=q)
    worker.start()

    time.sleep(600)
    _write_out_for_submit(fout=fout, input_que=q)
    worker.join()

    print('Finish predict on eval set in %s secs' % (time.time() - start_t))


if __name__ == '__main__':
    # script_for_eval()
    script_for_submit()
