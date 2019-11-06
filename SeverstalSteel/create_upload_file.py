import os
import sys
import pathlib
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.keras as keras

from MLBOX.Database.dataset import DataBase

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

from baseline import get_unet  # noqa: E402
from resunet import get_resunet, get_resunet_sigmoid  # noqa: E402
from create_database import SEGFORMAT  # noqa: E402
from variables import test_database  # noqa: E402


def _mask2rle(prediction) -> str:
    """A helper function turns model prediction into submission format

    Code borrowed from:
        https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

    Args:
        prediction:
            0-1 valued numpy array of prediction,
            1 - mask, 0 - background
    Returns:
        string in submission format
    """
    pixels = prediction.T.flatten()  # expand prediction in row-first order

    # compare flattened pixels with itself shifting by 1 pixel
    # the result is booleans marking the value swtich pts (from 1->0 or 0->1)
    runs = np.concatenate([[0], pixels, [0]])
    runs = runs[1:] != runs[:-1]
    value_switch_pts = np.where(runs)[0] + 1  # competition format is 1-indexed
    value_switch_pts[1::2] -= value_switch_pts[::2]
    return " ".join(str(x) for x in value_switch_pts)


def create_upload(model: keras.Model, database: DataBase,  out_f: str):
    """Create model predictions on database in submission format

    Args:
        model (keras.Model):
            the model to make predictions,
            model.predict(img) should tensor of shape (batch, 256, 1600, 5)
            where the last dimension is the probability of classes,
            which:
                0 -> background
                1~4 -> defect type 1-4
        database (DataBase):
            the dataset to predict on
        out_f (str):
            the output file to write.
            the file will be created and written.
    """
    results = {}
    with open(out_f, "w") as out_f:
        for data, label in database.get_input_tensor(epoch=1, batchsize=1):
            img = data["data"]
            img_id = data["dataid"]

            predictions = model.predict(img)
            maximum = tf.expand_dims(tf.reduce_max(predictions, axis=-1), axis=-1)
            # predictions = tf.cast(tf.equal(predictions, maximum), tf.int32)
            predictions = tf.cast(tf.greater_equal(predictions, tf.maximum(maximum, 0.5)), tf.int32)

            img_id = img_id.numpy()[0].decode()

            # for class_id in range(0, 2):
            #     img_cls_id = img_id + ".jpg_" + str(class_id)
            #     pred = predictions[..., class_id].numpy()
            #     encoded_pixels = _mask2rle(pred)
            #     if encoded_pixels:
            #         print(img_cls_id, encoded_pixels)
            #     results[img_cls_id] = encoded_pixels

            for class_id in range(1, 5):
                img_cls_id = img_id + ".jpg_" + str(class_id)

                pred = predictions[..., class_id-1].numpy()
                encoded_pixels = _mask2rle(pred)
                if encoded_pixels:
                    print(img_cls_id, encoded_pixels)
                results[img_cls_id] = encoded_pixels

        out_f.write("ImageId_ClassId,EncodedPixels\n")
        for key in sorted(results.keys()):
            out_f.write(key + "," + results[key] + "\n")


def _parse_csv(csv: str) -> dict:
    with open(str(csv), "r") as f:
        f.readline()  # skip header

        results = {}
        for line in f.readlines():
            imgid, result = line.split(",")
            results[imgid] = result.strip()

    return results


def _compare_two_csv_equal(csv1: str, csv2: str):
    result1 = _parse_csv(csv1)
    result2 = _parse_csv(csv2)

    p1 = pathlib.Path(csv1)
    p2 = pathlib.Path(csv2)
    imgids1 = set(result1.keys())
    imgids2 = set(result2.keys())
    print("Img ids in {} - Img ids in {} : ".format(p1.name, p2.name), imgids1 - imgids2)
    print("Img ids in {} - Img ids in {} : ".format(p2.name, p1.name), imgids2 - imgids1)

    union = imgids1 | imgids2
    assert len(union)
    non_empty = False
    for imgid in union:
        if not non_empty:
            if result1[imgid]:
                non_empty = True

        if result1[imgid] != result2[imgid]:
            print("Id: {} are different".format(imgid))
            print(result1[imgid])
            print(result2[imgid])
    assert non_empty


if __name__ == "__main__":
    # # model_path = "/archive/Steel/models/resunet_dice/model_014_0.11645.h5"
    # model_path = "/archive/Steel/models/resunet_bce/model_013_0.02174.h5"

    # # out_file = "/archive/Steel/models/unet_dice/unet_dice.csv"
    # # out_file = "/archive/Steel/models/resunet_dice/unet_dice.csv"
    # # out_file = "/archive/Steel/models/resunet_dice/resunet_dice.csv"
    # out_file = "/archive/Steel/models/resunet_bce/resunet_bce.csv"

    # # model = get_unet(softmax_activate=True)
    # # model = get_resunet(softmax_activate=True)
    # model = get_resunet_sigmoid(sigmoid_activate=True, n_class=4)
    # model.load_weights(model_path)

    # db = DataBase(formats=SEGFORMAT())
    # db.load_path(test_database)
    # db.config_parser(n_class=5)

    # create_upload(model=model, database=db, out_f=out_file)

    tf20 = "/archive/Steel/models/resunet_bce/resunet_bce.csv"
    tf114 = "/archive/Steel/models/resunet_bce/submission.csv"
    _compare_two_csv_equal(tf20, tf114)
