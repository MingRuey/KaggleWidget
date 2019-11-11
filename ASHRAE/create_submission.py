import os
import time
import pathlib

import numpy as np
import pandas

from ASHRAE.constants import train_meter
from ASHRAE.constants import sample_submission as sample
from ASHRAE.dnn_baseline import get_basemodel
from ASHRAE.create_database import get_val_dataset
from ASHRAE.create_database import get_test_dataset


def create_submission(models, output_csv):
    start = time.time()
    ds = get_test_dataset()
    print("finish import meter files", time.time() - start)

    output = pandas.read_csv(sample)
    output["meter_reading"] = output["meter_reading"].astype(np.float32)
    length = 0
    for feature, row_id in ds.get_dataset(1, 131072):
        row_id = row_id.numpy()

        pred = (models[0].predict(feature))[:, 0]
        pred = np.expm1(pred)
        for model in models[1:]:
            p = (model.predict(feature))[:, 0]
            p = np.expm1(p)
            pred = pred + p
        pred = pred / len(models)

        length += pred.shape[0]
        output.iloc[row_id, 1] = pred

    print(
        "finish prediction: ", time.time() - start,
        "with ", length, "data rows"
    )

    output.to_csv(output_csv, index=False)
    print("--- Finish create {} at {} ---".format(output_csv, time.time()-start))


def create_local_cv(models, output_csv):
    start = time.time()
    ds = get_val_dataset()
    print("finish import meter files", time.time() - start)

    output = pandas.read_csv(train_meter)
    output = pandas.DataFrame(
        {"row_id": np.arange(len(output))}
    )
    output["meter_reading"] = np.zeros(len(output))

    length = 0
    for feature, row_id in ds.get_dataset(1, 131072):
        row_id = row_id.numpy()

        pred = (models[0].predict(feature))[:, 0]
        pred = np.expm1(pred)
        for model in models[1:]:
            p = (model.predict(feature))[:, 0]
            p = np.expm1(p)
            pred = pred + p
        pred = pred / len(models)

        length += pred.shape[0]
        output.iloc[row_id, 1] = pred

    print(
        "finish prediction: ", time.time() - start,
        "with ", length, "data rows"
    )

    output.to_csv(output_csv, index=False)
    print("--- Finish create {} at {} ---".format(output_csv, time.time()-start))


if __name__ == "__main__":
    model_files = [
        "/archive/ASHRAE/models/dnn_base_iter-months/dnn_base-first_iter_months/model_002_1.23587.h5",
        "/archive/ASHRAE/models/dnn_base_iter-months/dnn_base-last_iter_months/model_039_1.22029.h5"
    ]
    # output_csv = "/archive/ASHRAE/models/dnn_base_iter-months/dnn_base_iter-months.csv"
    output_csv = "/archive/ASHRAE/models/dnn_base_iter-months/local_cv.csv"

    models = []
    for file in model_files:
        model = get_basemodel()
        model.load_weights(file)
        models.append(model)

    # create_submission(models, output_csv)
    create_local_cv(models, output_csv)