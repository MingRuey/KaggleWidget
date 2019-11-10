import pathlib
import pandas

from ASHRAE.constants import train_meter, train_weather
from ASHRAE.constants import buildings_meta
from ASHRAE.constants import per_meter, per_site_weather

from ASHRAE.create_database import _get_buildings_sites_map
from ASHRAE.create_database import AshraeDS
from ASHRAE.create_database import get_dataset, get_test_dataset


class TestAshraeDS:

    def test_columns_and_content(self):
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
            print(site_csv, meter_csv)
            ds.add_meter(weather_csv=str(site_csv), meter_csv=str(meter_csv))

    def test_train_test_split(self):
        ds = get_dataset(100)
        print(ds.data_count)

        train_ds, test_ds = ds.train_test_split(0.2)
        print(train_ds.data_count)
        print(test_ds.data_count)

        train_times = set()
        for feature, label in train_ds.get_dataset(1, 1):
            bid, meter = feature["building_id"], feature["meter"]
            month, day, hour = feature["month"], feature["day"], feature["hour"]
            train_times.add(
                (int(bid.numpy()), int(meter.numpy()), int(month.numpy()), int(day.numpy()), int(hour.numpy()))
            )

        test_times = set()
        for feature, label in test_ds.get_dataset(1, 1):
            bid, meter = feature["building_id"], feature["meter"]
            month, day, hour = feature["month"], feature["day"], feature["hour"]
            test_times.add(
                (int(bid.numpy()), int(meter.numpy()), int(month.numpy()), int(day.numpy()), int(hour.numpy()))
            )

        intersection = train_times & test_times
        print("intersection", len(intersection))

    def test_by_column_split(self):
        ds = get_dataset(3)
        print(ds.data_count)

        train_ds, test_ds = ds.train_test_split(
            0.2, target_column="last_half_months"
        )
        print(train_ds.data_count)
        print(test_ds.data_count)

        train_times = set()
        for feature, label in train_ds.get_dataset(1, 1):
            month = feature["month"].numpy()[0]
            train_times.add(month)

        test_times = set()
        for feature, label in test_ds.get_dataset(1, 1):
            month = feature["month"].numpy()[0]
            test_times.add(month)

        print(train_times)
        print(test_times)

    def test_testdataset(self):
        ds = get_test_dataset(3)
        print(ds.data_count)

        for feature, row_id in ds.get_dataset(1, 1):
            print(feature)
            print(row_id)
            break


class TestEmbedding:

    def test_weather(self):
        ds = get_test_dataset(3)
        for feature, row_id in ds:
            print(row_id, feature)


if __name__ == "__main__":
    test = TestAshraeDS()
    # test.test_train_test_split()
    # test.test_by_column_split()
    test.test_testdataset()
