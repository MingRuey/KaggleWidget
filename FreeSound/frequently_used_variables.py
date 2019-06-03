import pathlib
import logging
from MLBOX.Database.formats import TSFORMAT
from MLBOX.Database.dataset import DataBase

TRAIN_CURATED = "/rawdata/FreeSound/train_curated"
TRAIN_CURATED_CSV = "/rawdata/FreeSound/train_curated.csv"
TRAIN_NOISY = "/rawdata/FreeSound/train_noisy"
TRAIN_NOISY_CSV = "/rawdata/FreeSound/train_noisy.csv"
SAMPLE_SUBMISSION_CSV = "/rawdata/FreeSound/sample_submission.csv"


with open(SAMPLE_SUBMISSION_CSV, "r") as f:
    labels = f.readline().strip()
    labels = labels.split(",")[1:]

label_map = {}
for index, label in enumerate(labels):
    label_map[label] = index

CURATED_MAP = {}
with open(TRAIN_CURATED_CSV, "r") as f:
    f.readline()  # skip first line
    for line in f:
        filename, labels = line.split(",", 1)
        labels = labels.strip().strip("\"").split(",")
        CURATED_MAP[filename] = [label_map[label] for label in labels]

NOISY_MAP = {}
with open(TRAIN_NOISY_CSV, "r") as f:
    f.readline()  # skip first line
    for line in f:
        filename, labels = line.split(",", 1)
        labels = labels.strip().strip("\"").split(",")
        NOISY_MAP[filename] = [label_map[label] for label in labels]


train_noisy_tfrecords = "/archive/FreeSound/database/train_noisy"
train_curated_tfrecords = "/archive/FreeSound/database/train_curated"

DB_CURATED = DataBase(formats=TSFORMAT)
DB_CURATED.add_files(pathlib.Path(train_curated_tfrecords).glob("*.tfrecord"))
CURATED_DATA_COUNT = 4970

DB_ALL = DataBase(formats=TSFORMAT)
DB_ALL.add_files(pathlib.Path(train_noisy_tfrecords).glob("*.tfrecord"))
DB_ALL.add_files(pathlib.Path(train_curated_tfrecords).glob("*.tfrecord"))
ALL_DATA_COUNT = 24785
