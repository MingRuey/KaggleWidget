import os
import time
import pathlib

import numpy as np
import pandas

from ASHRAE.constants import sample_submission as sample
from ASHRAE.dnn_baseline import get_basemodel
from ASHRAE.create_database import get_test_dataset


if __name__ == "__main__":
    # model_file = "/archive/ASHRAE/models/test_run/tmp/model_001_0.71384.h5"
    # output_csv = "/archive/ASHRAE/models/test_run/submission-001-0.71384.csv"
    model_file = "/archive/ASHRAE/models/test_run/model_053_0.59484.h5"
    output_csv = "/archive/ASHRAE/models/test_run/submission-053_0.59484.csv"

    model = get_basemodel()
    model.load_weights(model_file)

    start = time.time()
    ds = get_test_dataset()
    print("finish import meter files", time.time() - start)

    output = pandas.read_csv(sample)
    output["meter_reading"] = output["meter_reading"].astype(np.float32)
    length = 0
    for feature, row_id in ds.get_dataset(1, 131072):
        row_id = row_id.numpy()
        pred = model.predict((feature))[:, 0]
        pred = np.expm1(pred)
        length += pred.shape[0]
        output.iloc[row_id, 1] = pred

    print(
        "finish prediction: ", time.time() - start,
        "with ", length, "data rows"
    )

    output.to_csv(output_csv, index=False)
    print("--- Finish create {} at {} ---".format(output_csv, time.time()-start))
