import os
import pathlib
import logging
from scipy.io import wavfile

file = os.path.basename(__file__)
file = pathlib.Path(file).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(message).1000s ',
    handlers=[logging.FileHandler("{}.log".format(file))]
    )

import tensorflow as tf  # noqa: E402
from MLBOX.Database.formats import TSFORMAT   # noqa: E402
from MLBOX.Database.formats import _tffeature_bytes   # noqa: E402
from MLBOX.Database.formats import _tffeature_float, _tffeature_int64   # noqa: E402
from MLBOX.Database.dataset import DataBase   # noqa: E402
from frequently_used_variables import TRAIN_CURATED, TRAIN_CURATED_CSV  # noqa: E402
from frequently_used_variables import TRAIN_NOISY, TRAIN_NOISY_CSV  # noqa: E402
from frequently_used_variables import label_map  # noqa: E402
from frequently_used_variables import CURATED_MAP, NOISY_MAP  # noqa: E402

IDMAP = {**CURATED_MAP, **NOISY_MAP}


def wav_reader(file, file_map=IDMAP):
    file = pathlib.Path(file)
    if not file.is_file():
        raise ValueError("Invalid file path")

    if file.suffix != ".wav":
        raise ValueError("Invalid extension")

    dataid = file.stem
    labels = file_map[str(file.name)]
    extension = file.suffix.strip(".")

    rate, array = wavfile.read(str(file))

    fields = {
        'filename': _tffeature_bytes(bytes(dataid, 'utf8')),
        'extension': _tffeature_bytes(bytes(extension, 'utf')),
        'encoded': _tffeature_int64(array),
        'length': _tffeature_int64(len(array)),
        'rate': _tffeature_int64(rate),
        'class': _tffeature_int64(list(labels)),
    }
    return tf.train.Example(features=tf.train.Features(feature=fields))


def create_db(in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

        db = DataBase(formats=TSFORMAT)
        db.build_database(
            reader=wav_reader,
            input_dir=in_dir,
            output_dir=out_dir
            )
    else:
        msg = "Output dir already exist: {}"
        logging.info(msg.format(out_dir))


if __name__ == "__main__":
    create_db(
        in_dir=TRAIN_CURATED,
        out_dir="/archive/FreeSound/database/train_curated"
    )
    create_db(
        in_dir=TRAIN_NOISY,
        out_dir="/archive/FreeSound/database/train_noisy"
    )
