import pathlib
import pandas

from ASHRAE.constants import train_meter, train_weather
from ASHRAE.constants import buildings_meta
from ASHRAE.constants import per_meter, per_site_weather

from ASHRAE.create_database import _get_buildings_sites_map
from ASHRAE.create_database import AshraeDS


def _get_ds(meter_files_to_add: int) -> AshraeDS:
    files_to_add = int(meter_files_to_add)

    bs_mapping = _get_buildings_sites_map()
    sites = {}
    for csv in pathlib.Path(per_site_weather).glob("*.csv"):
        site_id = int(csv.stem.split("-")[-1])
        sites[site_id] = csv

    ds = AshraeDS()
    cnt = 0
    for meter_csv in pathlib.Path(per_meter).glob("*.csv"):
        b_id = int(meter_csv.stem.split("-")[1])
        site_id = bs_mapping[b_id]
        site_csv = sites[site_id]
        ds.add_meter(weather_csv=str(site_csv), meter_csv=str(meter_csv))
        cnt += 1
        if cnt == files_to_add:
            break
    return ds


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
        ds = _get_ds(1000)
        print(ds.data_count)

        train_ds, test_ds = ds.train_test_split(0.2)
        print(train_ds.data_count)
        print(test_ds.data_count)
        for feature, label in train_ds.get_dataset(1, 1):
            print(feature.keys(), label)


class TestEmbedding:

    def test_weather(self):
        ds = _get_ds(3)


if __name__ == "__main__":
    test = TestAshraeDS()
    test.test_train_test_split()
