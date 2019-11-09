import os
import random
import time
import pathlib
import logging
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ASHRAE.constants import train_meter, train_weather
from ASHRAE.constants import test_meter, test_weather
from ASHRAE.constants import buildings_meta
from ASHRAE.constants import per_meter, per_site_weather
from ASHRAE.constants import test_per_meter, test_per_site


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


def create_per_meter_readings(base_dir):
    train = pandas.read_csv(train_meter)
    test = pandas.read_csv(test_meter)

    target = test
    grouped = target.groupby(["building_id", "meter"])

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


def create_per_site_weathers(base_dir):
    train = pandas.read_csv(train_weather)
    test = pandas.read_csv(test_weather)

    target_weather = test
    grouped = target_weather.groupby("site_id")

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

    def _random_pick_values_from_column(
            self,
            test_ratio: float,
            target_column: str,
            random_state: int = 42
            ):
        """random select subset of values from given column"""
        random.seed(random_state)

        #  only use first df to extract values,
        #  which may not hold (if there are unseen value in other dfs)
        df = self._dfs[0]
        values = set(df[target_column].values)

        n_sample = max(int(test_ratio*len(values)), 1)
        test_values = set(random.sample(values, n_sample))
        msg = "Extract elements {} from column '{}' be test split"
        print(msg.format(test_values, target_column))
        return test_values

    def train_test_split(
            self,
            test_ratio: float,
            target_column: str = None,
            random_state=42
            ):
        """Simple random split the data set

        Args:
            test_ratio (float): split ratio
            target_column (str, optional):
                the target column to stratify,
                if None, random split by added csv is used
            random_state (int, optional): the random seed

        Returns:
            a tuple of (train dataset, test dataset)
        """
        train_dfs = []
        test_dfs = []
        if target_column:
            if target_column == "first_half_months":
                test_values = {1, 2, 3, 4, 5, 6}
                target_column = "month"
            elif target_column == "last_half_months":
                test_values = {7, 8, 9, 10, 11, 12}
                target_column = "month"
            else:
                if target_column and any(target_column not in df for df in self._dfs):
                    msg = "Not recognized target column {}"
                    raise ValueError(msg.format(target_column))

                test_values = self._random_pick_values_from_column(
                    test_ratio=test_ratio,
                    target_column=target_column,
                    random_state=random_state
                )

            for df in self._dfs:
                index = df[target_column].isin(test_values)
                test_df = df.loc[index]
                train_df = df.loc[~index]
                train_dfs.append(train_df)
                test_dfs.append(test_df)

        elif len(self._dfs) == 1:
            print("Single csv is used, split over rows")
            train_df, test_df = train_test_split(
                self._dfs[0], test_size=test_ratio,
                random_state=random_state
            )
            train_dfs.append(train_df)
            test_dfs.append(test_df)
        else:
            print("Multi csvs are used, split over csvs")
            random.seed(random_state)

            n_sample = int(test_ratio * len(self._dfs))
            if n_sample < 1 or len(self._dfs) - n_sample < 1:
                msg = "Either test or train set is empty after {} test-split"
                raise ValueError(msg.format(test_ratio))

            random.shuffle(self._dfs)
            test_dfs.extend(self._dfs[:n_sample])
            train_dfs.extend(self._dfs[n_sample:])

            sampled_buildings = set(
                df.ix[0, "building_id"] for df in test_dfs
            )
            msg = "Extract buildings {} be test split"
            print(msg.format(sampled_buildings))

        return (
            AshraeDS(meta=self._meta.copy(), dfs=train_dfs),
            AshraeDS(meta=self._meta.copy(), dfs=test_dfs)
        )

    @staticmethod
    def _parser(feature):
        """Doing preprocess stuff like one-hot, ..."""
        if "meter_reading" in feature:
            # return meter reading at train
            label = feature.pop("meter_reading")
        else:
            # return row id at test
            label = feature.pop("row_id")

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
        merge_df = merge_df.sample(frac=1).reset_index(drop=True)  # fast shuffle

        dataset = tf.data.Dataset.from_tensor_slices(dict(merge_df))
        dataset = dataset.map(
            self._parser,
            num_parallel_calls=OUTPUT_PARALLEL_CALL)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batchsize, drop_remainder=False)
        dataset = dataset.prefetch(OUTPUT_BUFFER_TO_BATCH_RATIO)
        return dataset


def get_dataset(meter_files_to_add: int=-1):
    ds = AshraeDS()

    bs_mapping = _get_buildings_sites_map()
    sites = {}
    for csv in pathlib.Path(per_site_weather).glob("*.csv"):
        site_id = int(csv.stem.split("-")[-1])
        sites[site_id] = csv

    cnt = 0
    for meter_csv in pathlib.Path(per_meter).glob("*.csv"):
        b_id = int(meter_csv.stem.split("-")[1])
        site_id = bs_mapping[b_id]
        site_csv = sites[site_id]
        ds.add_meter(weather_csv=str(site_csv), meter_csv=str(meter_csv))
        cnt += 1
        if cnt == meter_files_to_add:
            break

    return ds


def get_test_dataset(meter_files_to_add: int=-1):
    ds = AshraeDS()

    bs_mapping = _get_buildings_sites_map()
    sites = {}
    for csv in pathlib.Path(test_per_site).glob("*.csv"):
        site_id = int(csv.stem.split("-")[-1])
        sites[site_id] = csv

    cnt = 0
    for meter_csv in pathlib.Path(test_per_meter).glob("*.csv"):
        b_id = int(meter_csv.stem.split("-")[1])
        site_id = bs_mapping[b_id]
        site_csv = sites[site_id]
        ds.add_meter(weather_csv=str(site_csv), meter_csv=str(meter_csv))
        cnt += 1
        if cnt == meter_files_to_add:
            break

    return ds


if __name__ == "__main__":
    create_per_meter_readings(base_dir=test_per_meter)
    # create_per_site_weathers(base_dir=test_per_site)
