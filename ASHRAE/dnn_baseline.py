import os
import sys
import pathlib
import argparse
import logging

import pandas
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow import feature_column

from MLBOX.Trainers.basetrainer import KerasBaseTrainner

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

from ASHRAE.create_database import AshraeDS  # noqa: E402
from ASHRAE.create_database import get_dataset  # noqa: E402


def _turn_on_log():

    file = os.path.basename(__file__)
    file = pathlib.Path(file).stem
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(name)s-%(message).1000s ',
        handlers=[logging.FileHandler("{}.log".format(file))]
    )


def get_basemodel():
    weather_columns = []
    weather_inputs = {}
    for name in [
            "air_temperature", "cloud_coverage", "dew_temperature",
            "precip_depth_1_hr", "sea_level_pressure", "wind_direction",
            "wind_speed"]:
        weather_columns.append(feature_column.numeric_column(name))
        weather_inputs[name] = keras.Input(shape=(1,), name=name)

    # building-related columns
    primary_use = feature_column.categorical_column_with_vocabulary_list(
        "primary_use", [
            "Religious worship", "Public services", "Education",
            "Entertainment/public assembly", "Healthcare",
            'Manufacturing/industrial', "Services", "Parking",
            "Office", "Warehouse/storage", "Retail", "Utility",
            "Food sales and service", "Technology/science",
            "Lodging/residential", "Other"
        ]
    )
    primary_use = feature_column.embedding_column(primary_use, 2)
    site_id = feature_column.categorical_column_with_identity("site_id", 16)
    site_id = feature_column.embedding_column(site_id, 2)
    building_id = feature_column.categorical_column_with_identity("building_id", 1449)
    building_id = feature_column.embedding_column(building_id , 2)
    meter = feature_column.categorical_column_with_identity("meter", 4)
    meter = feature_column.embedding_column(meter, 1)

    building_meta = [primary_use, site_id, building_id, meter]
    building_meta_inputs = {
        name: keras.Input(shape=(1,), name=name, dtype=tf.string) for name in
        ["primary_use"]
    }
    building_meta_inputs.update({
        name: keras.Input(shape=(1,), name=name, dtype=tf.int32) for name in
        ["site_id", "building_id", "meter"]
    })
    for name in ["square_feet", "year_built", "floor_count"]:
        building_meta.append(feature_column.numeric_column(name))
        building_meta_inputs[name] = keras.Input(shape=(1,), name=name)

    # time-related columns
    month = feature_column.categorical_column_with_identity("month", 12)
    month = feature_column.embedding_column(month, 1)
    day = feature_column.categorical_column_with_identity("day", 31)
    day = feature_column.embedding_column(day, 1)
    weekday = feature_column.categorical_column_with_identity("weekday", 7)
    weekday = feature_column.embedding_column(weekday, 1)
    hour = feature_column.categorical_column_with_identity("hour", 24)
    hour = feature_column.embedding_column(hour, 1)

    time_columns = [month, day, weekday, hour]
    time_inputs = {
        name: keras.Input(shape=(1,), name=name, dtype=tf.int32) for name in
        ["month", "day", "weekday", "hour"]
    }

    weather_layer = keras.layers.DenseFeatures(weather_columns)(weather_inputs)
    building_layer = keras.layers.DenseFeatures(building_meta)(building_meta_inputs)
    time_layer = keras.layers.DenseFeatures(time_columns)(time_inputs)

    weather_layer = keras.layers.Dense(
        50, activation="sigmoid", kernel_initializer="he_normal"
    )(weather_layer)
    building_layer = keras.layers.Dense(
        50, activation="sigmoid", kernel_initializer="he_normal"
    )(building_layer)
    time_layer = keras.layers.Dense(
        50, activation="sigmoid", kernel_initializer="he_normal"
    )(time_layer)

    concat = keras.layers.concatenate([
        weather_layer, building_layer, time_layer
    ], axis=-1)

    output = keras.layers.Dense(
        100, activation="sigmoid", kernel_initializer="he_normal"
    )(concat)

    output = keras.layers.Dense(
        1, activation="relu", kernel_initializer="he_normal"
    )(output)

    return keras.Model(
        inputs=[weather_inputs, building_meta_inputs, time_inputs],
        outputs=output
    )


def RMLSE(y_true, y_pred):
    mask = K.cast(K.greater(y_true, 1e-3), tf.float32)
    valid_meter_reading = mask * K.sqrt(
        K.mean(K.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred)))
    )
    # 4.0 is mean log1p value of all meter readings
    empty_meter_reading = (mask - 1) * K.sqrt(
        K.mean(K.square(4.0 - tf.math.log1p(y_pred)))
    )
    return valid_meter_reading + empty_meter_reading


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model architecture", type=str)
    parser.add_argument("output_path", help="where to store the models", type=str)
    parser.add_argument(
        "--augment", help="whether to augment the training data",
        action="store_true"
    )

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with strategy.scope():
        args = parser.parse_args()
        if args.model.lower() == "ann":
            model = get_basemodel()
            loss = RMLSE
        else:
            msg = "Unrecognized model type: {}"
            raise ValueError(msg.format(args.model))

        out_dir = pathlib.Path(args.output_path)
        if not out_dir.is_dir():
            print("Warning: Outputdir not exist, got {}".format(out_dir))
            print("Warning: ---Try Create Outputdir---")
            out_dir.mkdir(mode=0o775, parents=True)

        # _turn_on_log()

        db = get_dataset()
        train_db, vali_db = db.train_test_split(0.2)

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
            out_dir=str(out_dir)
        )

        trainner.train(
            train_db=train_db,  # should already config parser
            vali_db=vali_db,  # should already config parser
            lr_decay_factor=0.1,
            batch_size=256,
            min_epoch=1,
            max_epoch=300,
            early_stop_patience=30,
            load_best=True
        )
