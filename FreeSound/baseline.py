import os
import sys
import logging
import pathlib
import numpy as np

file = os.path.basename(__file__)
file = pathlib.Path(file).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(message).1000s ',
    handlers=[logging.FileHandler("{}.log".format(file))]
    )

import tensorflow as tf  # noqa: E402

tf.config.gpu.set_per_process_memory_growth(True)

from tensorflow.keras.layers import SimpleRNN, LSTM, Dense   # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit   # noqa: E402
from MLBOX.Trainers.TF.Keras_Callbacks import ModelLogger  # noqa: E402
# from MLBOX.Trainers.TF.Keras_Callbacks import LearningRateDecaySchedule  # noqa: E402
from MLBOX.Trainers.TF.Keras_Metrics import lwlrap  # noqa: E402
from frequently_used_variables import label_map  # noqa: E402
from frequently_used_variables import DB_CURATED as curated_db   # noqa: E402
from frequently_used_variables import CURATED_DATA_COUNT  # noqa: E402
from frequently_used_variables import DB_ALL as all_db  # noqa: E402
from frequently_used_variables import ALL_DATA_COUNT  # noqa: E402
from signal_transformer import NUM_MEL_BINS, ToMelParser, ToImgParser  # noqa: E402

NUM_OF_CLASS = len(label_map)


class BaseTrainner:

    def __init__(self, out_dir):
        if not pathlib.Path(str(out_dir)).is_dir():
            raise ValueError("Invalid output dir")
        self.out_dir = str(out_dir)
        self.tmp_dir = pathlib.Path(out_dir).joinpath("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(self.tmp_dir)

    def model(self):
        raise NotImplementedError()

    def train(
            self,
            train_decode_ops,
            vali_decode_ops,
            base_lr,
            database,
            number_of_data,
            train_vali_split_ratio=0.2,
            lr_decay_factor=0.5,
            batch_size=8,
            min_epoch=40,
            max_epoch=200,
            early_stop_patience=20,
            load_best=True
            ):

        optimizer = Adam(
            learning_rate=base_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False
            )
        # optimizer = SGD(learning_rate=base_lr)
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["binary_accuracy", lwlrap]
            )

        init_epoch = 0
        if load_best:
            weights = list(pathlib.Path(self.out_dir).glob("*.h5"))
            if weights:
                filename = weights[0].name
                ini_epoch = int(filename.split("_")[1])
                self.model.load_weights(str(weights))
                print("load pretrain weights from {}".format(weights))
                print("Re-train from epoch: {}".format(init_epoch))

        db = SimpleSplit(
                database,
                ratio_for_vallidation=train_vali_split_ratio
                )

        train_gener = db.get_train_dataset().get_input_tensor(
            decode_ops=train_decode_ops,
            num_of_class=NUM_OF_CLASS,
            epoch=max_epoch,
            batchsize=batch_size
        )

        vali_gener = db.get_vali_dataset().get_input_tensor(
            decode_ops=vali_decode_ops,
            num_of_class=NUM_OF_CLASS,
            epoch=max_epoch,
            batchsize=batch_size
        )

        self.model.fit_generator(
            initial_epoch=init_epoch,
            generator=train_gener,
            epochs=max_epoch,
            steps_per_epoch=int(
                num_of_data*(1-train_vali_split_ratio)
                ) // batch_size,
            callbacks=[
                ModelLogger(
                    temp_model_folder=self.tmp_dir,
                    best_model_folder=self.out_dir,
                    monitor='val_loss', verbose=1, mode='min'
                    ),
                ReduceLROnPlateau(
                    factor=lr_decay_factor,
                    patience=early_stop_patience // 2,
                    min_delta=1e-4,
                    cooldown=2,
                    min_lr=1e-6,
                    monitor='val_loss', verbose=1, mode='min',
                    ),
                TensorBoard(
                    log_dir=self.tmp_dir
                    ),
                EarlyStopping(
                    monitor='val_loss',
                    mode="min",
                    patience=early_stop_patience,
                    verbose=1
                    )
                ],
            validation_data=vali_gener,
            validation_steps=int(
                num_of_data*train_vali_split_ratio
                ) // batch_size,
            validation_freq=1
        )


class VanillaRNN(BaseTrainner):

    def __init__(self, units, *args, **kwargs):
        self.units = units
        self._model = None
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        if self._model is None:
            inputs = tf.keras.Input(
                shape=(None, NUM_MEL_BINS),
                name="data")
            rnn = SimpleRNN(
                units=self.units,
                activation="tanh",
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros'
                )(inputs)

            output = Dense(NUM_OF_CLASS, activation="sigmoid")(rnn)

            self._model = tf.keras.Model(inputs=inputs, outputs=output)
            print(self._model.summary())

        return self._model


class RawLSTM(BaseTrainner):

    def __init__(self, units, *args, **kwargs):
        self.units = units
        self._model = None
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        if self._model is None:
            inputs = tf.keras.Input(
                shape=(None, NUM_MEL_BINS),
                name="data"
            )
            lstm = LSTM(
                units=self.units,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                implementation=1
            )(inputs)

            output = Dense(NUM_OF_CLASS, activation="sigmoid")(lstm)
            self._model = tf.keras.Model(inputs=inputs, outputs=output)
            print(self._model.summary())

        return self._model


class RawRes(BaseTrainner):

    def __init__(self, units, *args, **kwargs):
        self.size = units
        self._model = None
        super().__init__(*args, **kwargs)

    @property
    def model(self):
        if self._model is None:
            inputs = tf.keras.Input(
                shape=(NUM_MEL_BINS, NUM_MEL_BINS, 1),
                name="data"
            )

            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_tensor=inputs,
                pooling="avg"
            )

            output = base_model.output
            output = Dense(NUM_OF_CLASS, activation="sigmoid")(output)

            self._model = tf.keras.Model(inputs=inputs, outputs=output)
            print(self._model.summary())

        return self._model


if __name__ == "__main__":
    model_options = {"vanilla": VanillaRNN, "lstm": RawLSTM, "resnet": RawRes}
    dataset_options = {
        "curated": (curated_db, CURATED_DATA_COUNT),
        "all": (all_db, ALL_DATA_COUNT)
        }
    augment_options = {
        "std": tf.image.per_image_standardization,
        "vertical": tf.image.random_flip_left_right,
        "horizontal": tf.image.random_flip_up_down,
        "contrast": tf.image.random_contrast,
        "bright": tf.image.random_brightness,
        "saturation": tf.image.random_saturation,
        "hue": tf.image.random_hue
    }

    helper = {
        "usage":
            "python3 baseline.py " +
            "[model options] [dataset option] [output path]" +
            "[augment option1] [augment option2] ...",
        "model options": list(model_options.keys()),
        "dataset options": list(dataset_options.keys()),
        "augment options": list(augment_options.keys())
        }

    if len(sys.argv) < 4:
        print("Must specify model, dataset and output dir")
        print("Usage: {}".format(helper["usage"]))
        helper.pop("usage")
        for key in helper.keys():
            print("Arg {}".format(key))
            print("    -- {}".format(helper[key]))
        sys.exit()

    model_fn = model_options.get(sys.argv[1].lower())
    model_name = sys.argv[1].lower()
    if model_fn is None:
        raise ValueError("Model must be one of {}".format(model_options))

    dataset = dataset_options.get(sys.argv[2].lower())
    if dataset is None:
        raise ValueError("Model must be one of {}".format(dataset_options))
    database = dataset[0]
    num_of_data = dataset[1]

    out_dir = pathlib.Path(sys.argv[3])
    if not out_dir.is_dir():
        print("Warning: Outputdir not exist, got {}".format(out_dir))
        print("Warning: ---Try Create Outputdir---")
        out_dir.mkdir(mode=0o775, parents=True)

    if model_name in ["vanilla", "lstm"]:
        model = model_fn(
            units=512,
            out_dir=str(out_dir)
            )

        model.train(
            base_lr=0.001,
            train_decode_ops=ToMelParser,
            vali_decode_ops=ToMelParser,
            database=database,
            number_of_data=num_of_data,
            batch_size=1
        )
    else:
        arg_ops = []
        if len(sys.argv) > 4:
            for arg in sys.argv[4:]:
                op = augment_options.get(arg)
                if op is None:
                    msg = "Not recognize augment option: {}"
                    raise ValueError(msg.format(arg))
                arg_ops.append(arg)

        def ImgParser(series):
            image = ToImgParser(ToMelParser(series))
            if "std" in arg_ops:
                op = augment_options.get("std")
                image = op(image)
            return image

        def AugImgParser(series):
            image = ToImgParser(ToMelParser(series))
            if arg_ops:
                for arg in arg_ops:
                    op = augment_options.get(arg)
                    image = op(image)
            return image

        model = model_fn(
            units=NUM_MEL_BINS,
            out_dir=str(out_dir),
            )

        model.train(
            base_lr=0.001,
            train_decode_ops=AugImgParser,
            vali_decode_ops=ImgParser,
            database=database,
            number_of_data=num_of_data,
            batch_size=8
        )
