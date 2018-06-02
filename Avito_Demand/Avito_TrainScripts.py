# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:32:19 2018
@author: MRChou

Code used for fast CNN model building.

Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from Avito_KerasPrototypes import av01_nn, av02_inceptv3
from KerasUtils import ImgLabelLoader, KerasModelTrainner

TRAIN_PATH = '/rawdata/Avito_Demand/imgs_train/'
PKL_PATH = '/archive/Avito/data_preprocess/'
TRAIN_PKL = 'train.pickle'
F01_TRAIN_PKL = 'F01_train.pickle'
F01_TEST_PKL = 'F01_test.pickle'
RANDOM_STATE = 42  # the random seed for sklearn train_test_split


def feature_scaling(df):
    target_cols = ['price', 'user_id_labelcount', 'title_labelcount',
                   'activation_date_month', 'activation_date_weekday',
                   'item_seq_number', 'image_top_1']
    scaler = MinMaxScaler()
    df[target_cols] = scaler.fit_transform(df[target_cols])
    return None


def random_split_train(pkl, test_size=0.2, scaling=True):
    """Split train by sklearn train_test_split with RANDOM_STATE."""
    df = pickle.load(open(PKL_PATH + pkl, 'rb'))

    if scaling:
        feature_scaling(df)

    # Split train, vali set
    train_set, vali_set = train_test_split(df,
                                           test_size=test_size,
                                           random_state=RANDOM_STATE
                                           )
    return train_set, vali_set


def cnn_get_loaders():
    train_set, vali_set = random_split_train(TRAIN_PKL,
                                             scaling=False)

    # Drop columns without images
    train_set.dropna(subset=['image'], inplace=True)
    vali_set.dropna(subset=['image'], inplace=True)

    # Combine image name with path:
    imgs_train = train_set['image'].apply(
        lambda file: TRAIN_PATH + file + '.jpg'
    )
    imgs_vali = vali_set['image'].apply(
        lambda file: TRAIN_PATH + file + '.jpg'
    )

    # Change deal probability into numpy array eachly
    label_train = train_set['deal_probability'].apply(
        lambda prob: numpy.array([prob])
    )
    label_vali = vali_set['deal_probability'].apply(
        lambda prob: numpy.array([prob])
    )

    train_loader = ImgLabelLoader(imgs=imgs_train.values,
                                  labels=label_train.values)
    vali_loader = ImgLabelLoader(imgs=imgs_vali.values,
                                 labels=label_vali.values)

    return train_loader, vali_loader


def train_nn():
    """A nn model trainner"""
    train_set, vali_set = random_split_train(F01_TRAIN_PKL,
                                             scaling=True
                                             )

    # Drop text feature
    train_set.drop(['item_id', 'description'], axis=1, inplace=True)
    vali_set.drop(['item_id', 'description'], axis=1, inplace=True)

    # let vali set matchs keras input
    vali_set = (vali_set.drop(['deal_probability'], axis=1).values,
                vali_set['deal_probability'].values
                )

    # load and train NN model
    model = av01_nn()
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop')
    model.summary()

    model.fit(x=train_set.drop(['deal_probability'], axis=1).values,
              y=train_set['deal_probability'].values,
              batch_size=1024,
              epochs=1,
              validation_data=vali_set,
              shuffle=True
              )

    model.save('av01_nn.h5')


def train_cnn():
    """A cnn model trainner"""
    train_loader, vali_loader = cnn_get_loaders()

    # CNN Model
    cnn = KerasModelTrainner(model=av02_inceptv3(),
                             model_name='av02_InceptionV3_0601',
                             train_loader=train_loader,
                             vali_loader=vali_loader
                             )

    cnn.compile(loss='mean_squared_error',
                optimizer='rmsprop'
                )

    cnn.fit(batch_size=128,
            epoch=1,
            augment=False,
            queue=10,
            log='av02_InceptionV3_0601.log')

    cnn.write_info()
    cnn.save()


def predict_test_cnn():
    model = '/home/mrchou/code/KaggleWidget/av02_InceptionV3_0601.h5'
    model = load_model(model)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.summary()
    batch_size = 128

    train_loader, vali_loader = cnn_get_loaders()

    train_steps = numpy.floor(train_loader.num_of_samples / batch_size)
    vali_steps = numpy.floor(vali_loader.num_of_samples / batch_size)

    train_gener = train_loader.batch_gener(batch_size=batch_size,
                                           epoch=1
                                           )
    vali_gener = vali_loader.batch_gener(batch_size=batch_size,
                                         epoch=1
                                         )

    print(model.evaluate_generator(train_gener,
                                   use_multiprocessing=True,
                                   steps=train_steps)
          )
    print(model.evaluate_generator(vali_gener,
                                   use_multiprocessing=True,
                                   steps=vali_steps)
          )


if __name__ == '__main__':
    predict_test_cnn()
