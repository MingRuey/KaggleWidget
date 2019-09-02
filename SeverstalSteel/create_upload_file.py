import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from MLBOX.Database.dataset import DataBase

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

from baseline import get_unet  # noqa: E402
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
            predictions = tf.cast(tf.equal(predictions, maximum), tf.int32)

            img_id = img_id.numpy()[0].decode()

            for class_id in range(1, 5):
                img_cls_id = img_id + ".jpg_" + str(class_id)

                pred = predictions[..., class_id].numpy()
                encoded_pixels = _mask2rle(pred)
                if encoded_pixels:
                    print(img_cls_id, encoded_pixels)
                results[img_cls_id] = encoded_pixels

        out_f.write("ImageId_ClassId,EncodedPixels\n")
        for key in sorted(results.keys()):
            out_f.write(key + "," + results[key] + "\n")

if __name__ == "__main__":
    model_path = "/archive/Steel/models/unet_dice/model_008_0.02904.h5"
    out_file = "/archive/Steel/models/unet_dice/unet_dice.csv"

    model = get_unet(softmax_activate=True)
    model.load_weights(model_path)

    db = DataBase(formats=SEGFORMAT())
    db.load_path(test_database)
    db.config_parser(n_class=5)

    create_upload(model=model, database=db, out_f=out_file)
