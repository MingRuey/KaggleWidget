import os
import sys
import pathlib
import logging
import argparse

import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
import tensorflow.keras.backend as K  # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.layers import Dense  # noqa: E402

from MLBOX.Models.TF.Keras.UNet import UNET, unet_loss, dice_loss, dice_coef  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402
from MLBOX.Trainers.basetrainer import KerasBaseTrainner  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit  # noqa: E402

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

from other_parsers import CLSFMT, CropClsFMT  # noqa: E402
from other_parsers2 import PerClassCropFMT  # noqa: E402
from variables import train_database, test_database  # noqa: E402


def get_classifier():
    inputs = keras.Input(shape=(256, 1600, 3))
    model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )
    dense = Dense(
        units=1,
        kernel_initializer="he_normal",
        activation="sigmoid"
    )(model.output)

    model = keras.Model(inputs=inputs, outputs=dense)
    return model


def xception():
    inputs = keras.Input(shape=(256, 400, 3))
    model = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )
    dense = Dense(
        units=4,
        kernel_initializer="he_normal",
        activation="sigmoid"
    )(model.output)

    model = keras.Model(inputs=inputs, outputs=dense)
    return model


def inception():
    inputs = keras.Input(shape=(256, 400, 3))
    model = keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )
    dense = Dense(
        units=4,
        kernel_initializer="he_normal",
        activation="sigmoid"
    )(model.output)

    model = keras.Model(inputs=inputs, outputs=dense)
    return model


def resnetv2():
    inputs = keras.Input(shape=(256, 400, 3))
    model = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )
    dense = Dense(
        units=4,
        kernel_initializer="he_normal",
        activation="sigmoid"
    )(model.output)

    model = keras.Model(inputs=inputs, outputs=dense)
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
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with strategy.scope():
        args = parser.parse_args()
        if args.model.lower() == "unet":
            raise ValueError("no resunet classfieir")
        elif args.model.lower() == "resunet":
            raise ValueError("No resunet classfieir")
        elif args.model.lower() == "resnet":
            model = get_classifier()
            fmt = CropClsFMT()
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif args.model.lower() == "resnetv2":
            model = resnetv2()
            fmt = PerClassCropFMT()
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif args.model.lower() == "xception":
            model = xception()
            fmt = PerClassCropFMT()
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif args.model.lower() == "inception":
            model = inception()
            fmt = PerClassCropFMT()
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            msg = "Unrecognized model type: {}"
            raise ValueError(msg.format(args.model))

        out_dir = pathlib.Path(args.output_path)
        if not out_dir.is_dir():
            print("Warning: Outputdir not exist, got {}".format(out_dir))
            print("Warning: ---Try Create Outputdir---")
            out_dir.mkdir(mode=0o775, parents=True)

        db = DataBase(formats=fmt)
        db.load_path(train_database)
        db = SimpleSplit(db, ratio_for_validation=0.2)
        train_db = db.get_train_dataset()
        vali_db = db.get_vali_dataset()
        train_db.config_parser(aug=True)
        vali_db.config_parser(aug=False)

        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad='False'
        )

        # thres = 0.5
        trainner = KerasBaseTrainner(
            model=model,
            loss=loss,
            optimizer=optimizer,
            # metrics=[
            #     "accuracy",
            #     tf.keras.metrics.FalseNegatives(thresholds=thres),
            #     tf.keras.metrics.FalsePositives(thresholds=thres),
            #     tf.keras.metrics.TrueNegatives(thresholds=thres),
            #     tf.keras.metrics.TruePositives(thresholds=thres)
            # ],
            out_dir=str(out_dir)
        )

        trainner.train(
            train_db=train_db,  # should already config parser
            vali_db=vali_db,  # should already config parser
            lr_decay_factor=0.1,
            batch_size=32,
            min_epoch=1,
            max_epoch=300,
            early_stop_patience=30,
            load_best=True
        )
