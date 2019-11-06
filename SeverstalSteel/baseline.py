import os
import sys
import pathlib
import logging
import argparse

import tensorflow as tf  # noqa: E402
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam

from MLBOX.Models.TF.Keras.UNet import UNET, unet_loss, dice_loss, dice_coef
from MLBOX.Database.formats import DataFormat, IMGFORMAT
from MLBOX.Database.dataset import DataBase
from MLBOX.Trainers.basetrainer import KerasBaseTrainner
from MLBOX.Scenes.SimpleSplit import SimpleSplit

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

from create_database import SEGFORMAT  # noqa: E402
from variables import train_database, test_database  # noqa: E402
from resunet import get_resunet, get_resunet_sigmoid  # noqa: E402


def _turn_on_log():

    file = os.path.basename(__file__)
    file = pathlib.Path(file).stem
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(name)s-%(message).1000s ',
        handlers=[logging.FileHandler("{}.log".format(file))]
    )


def get_unet(softmax_activate=False):
    inputs = tf.keras.layers.Input(shape=(256, 1600, 3), name="data")
    unet = UNET(n_base_filter=64, n_down_sample=4, n_class=5, padding="reflect")
    outputs = unet.forward(inputs)
    if softmax_activate:
        outputs = K.softmax(outputs, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model architecture", type=str)
    parser.add_argument("output_path", help="where to store the models", type=str)
    parser.add_argument(
        "--augment", help="whether to augment the training data",
        action="store_true"
    )

    # select model
    args = parser.parse_args()
    if args.model.lower() == "unet":
        model = get_unet(softmax_activate=True)
        loss = dice_loss
    elif args.model.lower() == "resunet":
        model = get_resunet(softmax_activate=True)
        # model = get_resunet_sigmoid(sigmoid_activate=True)
        loss = dice_loss

    else:
        msg = "Unrecognized model type: {}"
        raise ValueError(msg.format(args.model))

    out_dir = pathlib.Path(args.output_path)
    if not out_dir.is_dir():
        print("Warning: Outputdir not exist, got {}".format(out_dir))
        print("Warning: ---Try Create Outputdir---")
        out_dir.mkdir(mode=0o775, parents=True)

    # _turn_on_log()

    db = DataBase(formats=SEGFORMAT())
    db.load_path(train_database)
    db = SimpleSplit(db, ratio_for_validation=0.2)
    train_db = db.get_train_dataset()
    vali_db = db.get_vali_dataset()
    train_db.config_parser(n_class=5)
    vali_db.config_parser(n_class=5)

    # optimizer = SGD(
    #     learning_rate=0.005,
    #     momentum=0.8,
    #     nesterov=True
    # )

    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )

    trainner = KerasBaseTrainner(
        model=model,
        loss=loss,
        optimizer=optimizer,
        metrics=[dice_coef],
        out_dir=str(out_dir)
    )

    trainner.train(
        train_db=train_db,  # should already config parser
        vali_db=vali_db,  # should already config parser
        lr_decay_factor=0.1,
        batch_size=2,
        min_epoch=1,
        max_epoch=60,
        early_stop_patience=10,
        load_best=True
    )
