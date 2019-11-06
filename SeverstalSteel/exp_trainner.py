import os
import sys
import pathlib
import logging
import argparse

import tensorflow as tf  # noqa: E402
import tensorflow.keras.backend as K  # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402

from MLBOX.Models.TF.Keras.UNet import UNET, unet_loss, dice_loss, dice_coef  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402
from MLBOX.Trainers.basetrainer import KerasBaseTrainner  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit  # noqa: E402

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

from create_database import SEGFORMAT  # noqa: E402
from other_parsers import SEG_SINGLECLS_FMT, SEG_MASK_FMT  # noqa: E402
from variables import train_database, test_database  # noqa: E402
from resunet import get_resunet  # noqa: E402
from resunet import get_resunet_sigmoid  # noqa: E402


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
        raise ValueError("No Unet experiments")
    elif args.model.lower() == "resunet":
        model = get_resunet_sigmoid(sigmoid_activate=False, n_class=4)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        msg = "Unrecognized model type: {}"
        raise ValueError(msg.format(args.model))

    out_dir = pathlib.Path(args.output_path)
    if not out_dir.is_dir():
        print("Warning: Outputdir not exist, got {}".format(out_dir))
        print("Warning: ---Try Create Outputdir---")
        out_dir.mkdir(mode=0o775, parents=True)

    db = DataBase(formats=SEG_MASK_FMT())
    db.load_path(train_database)
    db = SimpleSplit(db, ratio_for_validation=0.2)
    train_db = db.get_train_dataset()
    vali_db = db.get_vali_dataset()
    train_db.config_parser(aug=True)
    vali_db.config_parser(aug=False)

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
        amsgrad='False'
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
        max_epoch=80,
        early_stop_patience=20,
        load_best=True
    )
