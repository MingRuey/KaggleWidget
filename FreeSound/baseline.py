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
from tensorflow.keras.optimizers import SGD   # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from MLBOX.Database.formats import TSFORMAT   # noqa: E402
from MLBOX.Database.dataset import DataBase   # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit   # noqa: E402
from MLBOX.Trainners.TF.Keras_Callbacks import ModelLogger   # noqa: E402
from MLBOX.Trainners.TF.Keras_Callbacks import LearningRateDecaySchedule  # noqa: E402
from KaggleWidget.FreeSound.create_database import label_map   # noqa: E402
from KaggleWidget.FreeSound.signal_transformer import TimeSeriesToMel   # noqa: E402
from KaggleWidget.FreeSound.signal_transformer import SpectrogramsToImage   # noqa: E402


TRAIN_NOISY = "/archive/FreeSound/database/train_noisy"
TRAIN_CURATED = "/archive/FreeSound/database/train_curated"
NUM_OF_CLASS = len(label_map)
TOTAL_DATA = 4970
SPLIT_RATIO = 0.1

db = DataBase(formats=TSFORMAT)
db.add_files(pathlib.Path(TRAIN_CURATED).glob("*.tfrecord"))
db = SimpleSplit(db, ratio_for_vallidation=SPLIT_RATIO)

NUM_MEL_BINS = 256
ToMel = TimeSeriesToMel(number_of_mel_bins=NUM_MEL_BINS)
ToMelParser = ToMel.get_parser()
ToImg = SpectrogramsToImage(NUM_MEL_BINS, NUM_MEL_BINS)
ToImgParser = ToImg.get_parser()


def ImgParser(series):
    return ToImgParser(ToMelParser(series))


class BaseTrainner:

    def __init__(
            self,
            base_lr,
            out_dir,
            min_epoch=20,
            max_epoch=100,
            early_stop_patience=8,
            decode_ops=ToMelParser
            ):
        if not pathlib.Path(str(out_dir)).is_dir():
            raise ValueError("Invalid output dir")
        self.out_dir = str(out_dir)

        self.tmp_dir = pathlib.Path(out_dir).joinpath("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(self.tmp_dir)

        self.base_lr = base_lr
        self.lr_decay_factor = 0.5
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.patience = early_stop_patience

        self.decode_ops = decode_ops

    def model(self):
        raise NotImplementedError()

    def train(self):

        optimizer = SGD(learning_rate=self.base_lr)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
            )

        train_gener = db.get_train_dataset().get_input_tensor(
            decode_ops=self.decode_ops,
            num_of_class=NUM_OF_CLASS,
            epoch=self.max_epoch,
            batchsize=1
        )

        vali_gener = db.get_train_dataset().get_input_tensor(
            decode_ops=self.decode_ops,
            num_of_class=NUM_OF_CLASS,
            epoch=self.max_epoch,
            batchsize=1
        )

        self.model.fit_generator(
            generator=train_gener,
            epochs=self.max_epoch,
            steps_per_epoch=int(TOTAL_DATA*(1-SPLIT_RATIO)),
            callbacks=[
                ModelLogger(
                    temp_model_folder=self.tmp_dir,
                    best_model_folder=self.out_dir,
                    monitor='val_loss', verbose=1, mode='min'
                    ),
                LearningRateDecaySchedule.step_decay_by_epoch(
                    decay=self.lr_decay_factor,
                    epochs_to_decay=self.min_epoch
                    ),
                TensorBoard(
                    log_dir=self.tmp_dir
                    ),
                EarlyStopping(
                    monitor='val_loss',
                    mode="min",
                    patience=self.patience,
                    verbose=1
                    )
                ],
            validation_data=vali_gener,
            validation_steps=int(TOTAL_DATA*SPLIT_RATIO),
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

            output = Dense(NUM_OF_CLASS, activation="softmax")(rnn)

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

            output = Dense(NUM_OF_CLASS, activation="softmax")(lstm)
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
            output = Dense(NUM_OF_CLASS, activation="softmax")(output)

            self._model = tf.keras.Model(inputs=inputs, outputs=output)
            print(self._model.summary())

        return self._model



if __name__ == "__main__":
    device_options = {"0": "0", "1": "1", "all": "0, 1"}
    model_options = {"vanilla": VanillaRNN, "lstm": RawLSTM, "resnet": RawRes}

    if len(sys.argv) < 4:
        raise ValueError("Must specify model, device options and output dir")

    device = device_options.get(sys.argv[1])
    if device is None:
        raise ValueError("Device must be one of {}".format(device_options))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    model_fn = model_options.get(sys.argv[2].lower())
    model_name = sys.argv[2].lower()
    if model_fn is None:
        raise ValueError("Model must be one of {}".format(model_options))

    out_dir = pathlib.Path(sys.argv[3])
    if not out_dir.is_dir():
        raise ValueError("Outputdir not exist, got {}".format(out_dir))

    if model_name in ["vanilla", "lstm"]:
        model = model_fn(
            units=512,
            base_lr=0.001,
            out_dir=str(out_dir)
            )
    else:
        model = model_fn(
            units=NUM_MEL_BINS,
            base_lr=0.01,
            out_dir=str(out_dir),
            decode_ops=ImgParser
            )

    model.train()
