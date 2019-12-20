import os
import sys
import pathlib
import argparse
import logging

import pandas
import numpy as np
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


def get_gru():
    inputs = keras.Input(shape=(None, 50))
    gru_cell = keras.layers.GRU(
        units=100, kernel_initializer="he_normal",
        return_sequences=True
    )(inputs)
    return keras.Model(inputs=inputs, outputs=gru_cell)


if __name__ == "__main__":
    model = get_gru()

    model.summary()

    series = np.ones((2, 50))
    pred = model.predict(series)
    print(pred.shape)
    print(pred)
