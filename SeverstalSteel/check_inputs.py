import os
import sys

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import tensorflow as tf  # noqa: E402
from MLBOX.Database.formats import _tffeature_bytes, _tffeature_float, _tffeature_int64  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402
from create_database import SEGFORMAT  # noqa: E402


if __name__ == "__main__":
    pDB_train = "/archive/Steel/database/train"

    db = DataBase(formats=SEGFORMAT())
    db.load_path(pDB_train)
    db.config_parser()

    for data, label in db.get_input_tensor(epoch=1, batchsize=1):
        img, img_id = data["data"], data["dataid"]
        print(img_id)
        print(img)
        print(label)
        break
