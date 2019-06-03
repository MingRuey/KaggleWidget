import sys
import pandas
import tensorflow as tf
from baseline import RawRes, RawLSTM
from MLBOX.Database.dataset import DataBase
from MLBOX.Scenes.SimpleSplit import SimpleSplit
from MLBOX.Trainers.TF.Keras_Callbacks import ModelLogger
from MLBOX.Trainers.TF.Keras_Metrics import lwlrap
from frequently_used_variables import label_map
from frequently_used_variables import DB_CURATED as curated_db
from frequently_used_variables import CURATED_DATA_COUNT
from frequently_used_variables import DB_ALL as all_db
from frequently_used_variables import ALL_DATA_COUNT
from signal_transformer import NUM_MEL_BINS, ImgParser


class ValidateOnDB:

    def __init__(
            self,
            model,
            database: DataBase,
            decode_ops,
            class_names
            ):
        """Create a ValidateOnDB scene

        Args:
            model: the target model to make prediction
            database: the target DB to validate on
            decode_ops: decode op for parsing database
            class_names: a list of class name string in order
        """
        if not isinstance(model, tf.keras.models.Model):
            msg = "{} currently support only tf.keras model"
            raise NotImplementedError(msg.format(self.__class__.__name__))

        self._model = model
        self._db = database
        self._decode_ops = decode_ops
        self._class_names = list(class_names)

    def create_predict_dataframe(self):

        gener = self._db.get_input_tensor(
            decode_ops=self._decode_ops,
            num_of_class=len(self._class_names),
            epoch=1,
            batchsize=1
        )

        preds = []
        for data, label in gener:
            data_id = data["dataid"].numpy()[0].decode()
            data_content = data["data"]
            pred = data_id, self._model.predict(data_content)
            preds.append(pred)

        preds.sort(key=lambda x: x[0])
        return pandas.DataFrame(preds, columns=["id", "features"])


if __name__ == "__main__":
    res_dir = "/archive/FreeSound/models/resnet_curated/model_031_0.03687.h5"
    model = RawRes(
        units=NUM_MEL_BINS,
        out_dir=""
    )
    model = model.model
    model.load_weights(res_dir)

    vali_db = SimpleSplit(
        curated_db, ratio_for_vallidation=0.2
        ).get_vali_dataset()

    scene = ValidateOnDB(
        model=model,
        database=vali_db,
        decode_ops=ImgParser,
        class_names=list(label_map.keys())
        )

    print(scene.create_predict_dataframe())
