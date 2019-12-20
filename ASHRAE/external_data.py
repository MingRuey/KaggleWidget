import pickle

import numpy as np
import pandas

from ASHRAE.constants import train_meter, train_weather
from ASHRAE.constants import test_meter, test_weather
from ASHRAE.constants import buildings_meta
from ASHRAE.constants import per_meter, per_site_weather
from ASHRAE.constants import val_per_meter
from ASHRAE.constants import test_per_meter, test_per_site

from ASHRAE.constants import site0_buildings
from ASHRAE.constants import site0_csv


def _check_csv_pkl_equal():
    with open(site0_pkl, "rb") as f:
        pkl = pickle.load(f)

    csv = pandas.read_csv(site0_csv)
    csv["timestamp"] = pandas.to_datetime(csv["timestamp"])

    print(pkl.columns)
    print(pkl.shape)
    print(csv.columns
    print(csv.shape)

    print(csv.timestamp.equals(pkl.timestamp))
    csv.drop("timestamp", axis=1, inplace=True)
    pkl.drop("timestamp", axis=1, inplace=True)

    for row in range(csv.shape[0]):
        csv_row = csv.iloc[row]
        pkl_row = pkl.iloc[row]
        if not np.all(np.isclose(csv_row.values, pkl_row.values, equal_nan=True)):
            print("Non equal rows", row)
            print(csv_row.values)
            print(pkl_row.values)
            print(csv_row.equals(pkl_row))
            break


if __name__ == "__main__":
    _check_csv_pkl_equal()
