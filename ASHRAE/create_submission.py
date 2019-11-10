import os
import time
import pathlib

import numpy as np
import pandas

from ASHRAE.constants import sample_submission as sample
from ASHRAE.dnn_baseline import get_basemodel
from ASHRAE.create_database import get_test_dataset


if __name__ == "__main__":
    model_files = [
        "/archive/ASHRAE/models/dnn_base-first-half-months/model_003_1.64434.h5",
        "/archive/ASHRAE/models/dnn_base-last-half-months/model_007_1.77415.h5"
    ]
    output_csv = "/archive/ASHRAE/models/dnn_base-last-half-months/model_half-months.csv"

    models = []
    for file in model_files:
        model = get_basemodel()
        model.load_weights(file)
        models.append(model)

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
