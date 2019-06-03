import sys
import pathlib
import numpy as np
from scipy.io import wavfile
from MLBOX.Database.formats import TSFORMAT
from MLBOX.Database.dataset import DataBase
from KaggleWidget.FreeSound.create_database import label_map
from KaggleWidget.FreeSound.signal_transformer import TimeSeriesToMel
from KaggleWidget.FreeSound.signal_transformer import SpectrogramsToImage


TRAIN_NOISY = "/archive/FreeSound/database/train_noisy"
TRAIN_CURATED = "/archive/FreeSound/database/train_curated"
NUM_OF_CLASS = len(label_map)
# print("{} labels: {}".format(NUM_OF_CLASS, label_map.keys()))

db = DataBase(formats=TSFORMAT)
db.add_files(pathlib.Path(TRAIN_CURATED).glob("*.tfrecord"))

feature_num = 224
to_mel = TimeSeriesToMel(number_of_mel_bins=feature_num)
to_mel = to_mel.get_parser()
to_img = SpectrogramsToImage(feature_num, feature_num)
to_img = to_img.get_parser()


def parser(series):
    return to_img(to_mel(series))


count = 0
for data, label in db.get_input_tensor(
        decode_ops=parser,
        num_of_class=NUM_OF_CLASS,
        epoch=1, batchsize=1
        ):

    dataid = np.array(data['dataid'])[0].decode()
    dataarr = np.array(data['data'])

    file = "/rawdata/FreeSound/train_curated/{}.wav".format(dataid)
    rate, array = wavfile.read(file)

    count += 1
    print(dataarr.shape, array.shape, label.shape)
