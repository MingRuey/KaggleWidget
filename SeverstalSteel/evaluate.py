import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam

from MLBOX.Database.dataset import DataBase
from MLBOX.Models.TF.Keras.UNet import UNET

from MLBOX.Database.formats import DataFormat, IMGFORMAT
from MLBOX.Scenes.SimpleSplit import SimpleSplit

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

from MLBOX.Models.TF.Keras.UNet import UNET, unet_loss, dice_coef  # noqa: E402
from create_database import SEGFORMAT  # noqa: E402
from baseline import get_unet  # noqa: E402
from variables import train_database, test_database  # noqa: E402


def dice_wrapper(softmax_activate=False):

    def loss(y_true, y_pred):
        if softmax_activate:
            y_pred = tf.nn.softmax(logits=y_pred, axis=-1)

        return dice_coef(y_true=y_true, y_pred=y_pred)

    return loss


def evaluate_on_dataset(model: keras.Model, database: DataBase):
    """Make model evaluate on dataset"""
    return model.evaluate(
        x=database.get_input_tensor(epoch=1, batchsize=1),
        steps=database.data_count
    )


def create_kaggle_outputs(model: keras.model, database: Database):
    """Create model prediction outputs on database"""
    model.predict()


if __name__ == "__main__":
    model_path = "/archive/Steel/models/unet/model_012_0.15096.h5"
    # model_path = "/archive/Steel/models/unet_dice/model_008_0.02904.h5"
    model = get_unet(softmax_activate=False)
    loss = dice_wrapper(softmax_activate=True)
    model.compile(optimizer=Adam(), loss=loss)
    model.load_weights(model_path)

    db = DataBase(formats=SEGFORMAT())
    db.load_path(train_database)
    db = SimpleSplit(db, ratio_for_validation=0.2)

    vali_db = db.get_vali_dataset()
    vali_db.config_parser(n_class=5)

    res = evaluate_on_dataset(model=model, database=vali_db)
    print("model: ", model_path, "result: ", res)
