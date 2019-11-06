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
from SeverstalSteel.other_parsers2 import PerClassCropFMT  # noqa: E402
from variables import train_database, test_database  # noqa: E402


if __name__ == "__main__":
    db = DataBase(formats=PerClassCropFMT())
    db.load_path(train_database)
    db = SimpleSplit(db, ratio_for_validation=0.2)
    train_db = db.get_train_dataset()
    vali_db = db.get_vali_dataset()
    train_db.config_parser(aug=True)
    vali_db.config_parser(aug=False)

    for img, label in train_db.get_dataset(1, 1):
        print(img.shape, label.shape)
        print(label)
