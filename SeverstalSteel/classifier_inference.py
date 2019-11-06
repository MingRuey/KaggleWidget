import os
import sys
import pathlib
import logging
import argparse

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
import tensorflow.keras.backend as K  # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.layers import Dense  # noqa: E402

from MLBOX.Models.TF.Keras.UNet import UNET, unet_loss, dice_loss, dice_coef  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402
from MLBOX.Trainers.basetrainer import KerasBaseTrainner  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit  # noqa: E402

from other_parsers import CLSFMT, CropInferenceFMT  # noqa: E402
from other_parsers2  import PerClsCropInferenceFMT  # noqa: E402
from variables import train_database, test_database  # noqa: E402
from classifier_trainner import get_classifier, get_perclass_classifier  # noqa: E402


def _get_model_score(model, image):
    image = image[0, ...]  # remove fake batch shape
    pred = model.predict(image)
    return max(pred)


def evalute_model(model, dataset, threshold=0.5):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    cnt = 0
    for img, label in dataset:
        label = label[0, ...].numpy()[0]
        pred = _get_model_score(model, img)[0] > threshold
        if label == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1
        cnt += 1
    print(" TP: ", tp, " FP: ", fp, " TN: ", tn, " FN: ", fn, " total: ", cnt)


def _perclass_model_score(model, image):
    image = image[0, ...]  # remove fake batch shape
    pred = model.predict(image)
    pred = tf.reduce_max(pred, axis=0)
    return pred


def _to_confusion_matrix(gt, pred):
    tp = tf.cast((gt == 1) & pred, dtype=tf.int32)
    fn = tf.cast((gt == 1) & tf.logical_not(pred), dtype=tf.int32)
    fp = tf.cast((gt == 0) & pred, dtype=tf.int32)
    tn = tf.cast((gt == 0) & tf.logical_not(pred), dtype=tf.int32)
    return tp, fn, fp, tn


def perclass_evaluate(model, dataset, threshold=0.5):
    tp = np.array([0, 0, 0, 0])
    fn = np.array([0, 0, 0, 0])
    fp = np.array([0, 0, 0, 0])
    tn = np.array([0, 0, 0, 0])
    cnt = 0
    for img, label in dataset:
        label = label[0]
        pred = _perclass_model_score(model, img) > threshold
        pred = _to_confusion_matrix(label, pred)
        tp = tp + pred[0].numpy()
        fn = fn + pred[1].numpy()
        fp = fp + pred[2].numpy()
        tn = tn + pred[3].numpy()
        cnt += 1

    print(" TP: ", tp)
    print(" FN: ", fn)
    print(" TN: ", tn)
    print(" FP: ", fp)
    print(" total: ", cnt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model architecture", type=str)
    parser.add_argument("model_path", help="the models file path", type=str)

    args = parser.parse_args()
    if args.model.lower() == "unet":
        raise ValueError("No resunet classfieir")
    elif args.model.lower() == "resunet":
        raise ValueError("No resunet classfieir")
    elif args.model.lower() == "resnet":
        model = get_classifier()
        fmt = CropInferenceFMT()
    elif args.model.lower() == "resnet-perclass":
        model = get_perclass_classifier()
        fmt = PerClsCropInferenceFMT()
    else:
        msg = "Unrecognized model type: {}"
        raise ValueError(msg.format(args.model))

    model_file = args.model_path
    if not pathlib.Path(model_file).is_file():
        raise ValueError("Invalid model file: {}".format(model_file))
    model.load_weights(model_file)

    db = DataBase(formats=fmt)
    db.load_path(train_database)
    db = SimpleSplit(db, ratio_for_validation=0.2)
    train_db = db.get_train_dataset()
    vali_db = db.get_vali_dataset()
    train_db.config_parser()
    vali_db.config_parser()

    print("Train data:")
    # evalute_model(model, train_db.get_dataset(1, 1))
    perclass_evaluate(model, train_db.get_dataset(1, 1))

    print("Validation data:")
    # evalute_model(model, vali_db.get_dataset(1, 1))
    perclass_evaluate(model, vali_db.get_dataset(1, 1))
