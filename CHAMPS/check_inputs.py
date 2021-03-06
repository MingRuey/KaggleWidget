from MLBOX.Database.dataset import DataBase
from create_database import FeaturesV0
from variables import train_features_v0, test_features_v0  # noqa: E402


if __name__ == "__main__":
    dataformat = FeaturesV0()
    db = DataBase(formats=dataformat)
    db.load_path(train_features_v0)
    db.config_parser()

    for data, label in db.get_input_tensor(epoch=1, batchsize=1):
        data = data["data"]
        for key in data.keys():
            print(key, ":  ", data[key])
        print(label)
        break
