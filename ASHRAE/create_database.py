import os
import time
import pathlib
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ASHRAE.constants import train_meter, train_weather
from ASHRAE.constants import buildings_meta
from ASHRAE.constants import per_meter, per_site_weather


OUTPUT_BUFFER_TO_BATCH_RATIO = 16
OUTPUT_PARALLEL_CALL = 16


def one_hot_time_data(value):
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt.year, dt.month, dt.day, dt.weekday(), dt.hour


def _get_buildings_sites_map():
    building_meta = pandas.read_csv(buildings_meta)

    # Sepearte meters by sites and buildings
    b_ids = building_meta.building_id.values
    s_ids = building_meta.site_id.values

    mapping = {
        b_id: s_id
        for b_id, s_id in zip(b_ids, s_ids)
    }
    return mapping


def check_time_interval():
    meter = pandas.read_csv(train_meter)
    mgrouped = meter.groupby(["building_id", "meter"])

    weather = pandas.read_csv(train_weather)
    wgrouped = weather.groupby("site_id")

    bs_mapping = _get_buildings_sites_map()

    site_weathers = {}
    for site_id, group in wgrouped:
        site_weathers[site_id] = group

    for gp_name, group in mgrouped:
        b_id, m_type = gp_name
        site_id = bs_mapping[b_id]
        site_group = site_weathers[site_id]

        print("site, site_len, building, meter, meter_len",
              site_id, site_group.shape, b_id, m_type, group.shape)

        meter_times = set(group["timestamp"])
        site_times = set(site_group["timestamp"])

        print("meter times, site_times:", len(meter_times), len(site_times))
        print("times in meter but not site", meter_times - site_times)
        print("times in sites but not in meteres", site_times - meter_times)


def create_per_meter_readings(base_dir=per_meter):
    train_meter = pandas.read_csv(train_meter)
    grouped = train_meter.groupby(["building_id", "meter"])

    base_dir = pathlib.Path(base_dir)
    if not base_dir.is_dir():
        raise ValueError("Invalid base dir {}".format(base_dir))

    fname = "building-{:04d}-meter-{:1d}.csv"
    for gp_name, group in grouped:
        b_id, m_type = gp_name

        # process datetime
        group[["year", "month", "day", "weekday", "hour"]] = \
            group.apply(lambda x: one_hot_time_data(
                x["timestamp"]), axis=1, result_type="expand")
        group.drop("timestamp", axis=1, inplace=True)

        file = fname.format(b_id, m_type)
        if base_dir.joinpath(file).is_file():
            print("file: {} already exist".format(file))
        group.to_csv(str(base_dir.joinpath(file)), index=False)


def create_per_site_weathers(base_dir=per_site_weather):
    train_weather = pandas.read_csv(train_weather)
    grouped = train_weather.groupby("site_id")

    base_dir = pathlib.Path(base_dir)
    if not base_dir.is_dir():
        raise ValueError("Invalid base dir {}".format(base_dir))

    fname = "site-{:02d}.csv"
    for site_id, group in grouped:
        # process datetime
        group[["year", "month", "day", "weekday", "hour"]] = \
            group.apply(lambda x: one_hot_time_data(
                x["timestamp"]), axis=1, result_type="expand")
        group.drop("timestamp", axis=1, inplace=True)

        file = fname.format(site_id)
        if base_dir.joinpath(file).is_file():
            print("file: {} already exist".format(file))

        group.to_csv(str(base_dir.joinpath(file)), index=False)


class AshraeDS:
    """Dataset class for ASHRAE competition"""

    def __init__(
            self,
            meta=None,
            dfs=None
            ):
        self._meta = meta if meta is not None else pandas.read_csv(buildings_meta)
        self._dfs = dfs if dfs is not None else []

    @property
    def data_count(self):
        if not self._dfs:
            return 0
        return sum(df.shape[0] for df in self._dfs)

    def add_meter(self, weather_csv: str, meter_csv: str = None):
        df = pandas.read_csv(weather_csv)
        if meter_csv:
            meter = pandas.read_csv(meter_csv)
            df = df.merge(
                right=meter, how="inner",
                validate="one_to_one"
            )
            df = df.merge(
                right=self._meta, how="left",
                validate="many_to_one"
            )

        # print(weather_csv, meter_csv)
        # with pandas.option_context("display.max_columns", None):
            # print(df.head())
        df.fillna(value=0, inplace=True)
        for c in df:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32", copy=False)
        self._dfs.append(df)

    def train_test_split(self, test_ratio: float, random_state=42):
        train_dfs = []
        test_dfs = []
        for df in self._dfs:
            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio, random_state=random_state
            )
            train_dfs.append(train_df)
            test_dfs.append(test_df)

        return (
            AshraeDS(meta=self._meta.copy(), dfs=train_dfs),
            AshraeDS(meta=self._meta.copy(), dfs=test_dfs)
        )

    @staticmethod
    def _parser(feature):
        """Doing preprocess stuff like one-hot, ..."""
        label = feature.pop("meter_reading")
        feature["day"] = feature["day"] - 1
        feature["month"] = feature["month"] - 1
        return feature, label

    def get_dataset(self, epoch: int, batchsize: int):
        """Retrieve the dataset object for tensorflow training"""
        if not self._dfs:
            raise RuntimeError("No meter is added")

        merge_df = self._dfs[0]
        if len(self._dfs) > 1:
            start = time.time()
            merge_df = merge_df.append(self._dfs[1:], ignore_index=True)
            print("It takes ", time.time() - start, "to append dataframes", len(self._dfs))
        merge_df = merge_df.sample(frac=1).reset_index(drop=True)  # fast shuffle

        dataset = tf.data.Dataset.from_tensor_slices(dict(merge_df))
        dataset = dataset.map(
            self._parser,
            num_parallel_calls=OUTPUT_PARALLEL_CALL)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batchsize, drop_remainder=True)
        dataset = dataset.prefetch(OUTPUT_BUFFER_TO_BATCH_RATIO)
        return dataset


def get_dataset():
    ds = AshraeDS()

    bs_mapping = _get_buildings_sites_map()
    sites = {}
    for csv in pathlib.Path(per_site_weather).glob("*.csv"):
        site_id = int(csv.stem.split("-")[-1])
        sites[site_id] = csv

    for meter_csv in pathlib.Path(per_meter).glob("*.csv"):
        b_id = int(meter_csv.stem.split("-")[1])
        site_id = bs_mapping[b_id]
        site_csv = sites[site_id]
        ds.add_meter(weather_csv=str(site_csv), meter_csv=str(meter_csv))

    return ds


if __name__ == "__main__":
    # create_per_meter_readings()
    # create_per_site_weathers()

    check_time_interval()
