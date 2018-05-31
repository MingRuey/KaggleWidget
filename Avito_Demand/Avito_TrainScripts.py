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
from Avito_KerasPrototypes import av01_nn, av02_inceptv3
from KerasUtils import ImgLabelLoader, KerasModelTrainner

TRAINIMG_PATH = '/rawdata/Avito_Demand/imgs_train/'
VALIIMG_PATH = '/rawdata/Avito_Demand/imgs_test/'
PKL_PATH = '/archive/Avito/data_preprocess/'
TRAIN_PKL = 'train.pickle'
F01_PKL = 'F01_train.pickle'
RANDOM_STATE = 42  # the random seed for sklearn train_test_split


def train_nn():
    df = pickle.load(open(PKL_PATH + F01_PKL, 'rb'))

    # Drop text feature
    df.drop(['item_id', 'description'], axis=1, inplace=True)

    # Feature Normarlization
    target_cols = ['price', 'user_id_labelcount', 'title_labelcount',
                   'activation_date_month', 'activation_date_weekday',
                   'item_seq_number', 'image_top_1']
    scaler = MinMaxScaler()
    df[target_cols] = scaler.fit_transform(df[target_cols])

    # Split train, vali set
    train_set, vali_set = train_test_split(df,
                                           test_size=0.2,
                                           random_state=RANDOM_STATE)
    vali_set = (vali_set.drop(['deal_probability'], axis=1).values,
                vali_set['deal_probability'].values
                )

    # NN model
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
    df = pickle.load(open(PKL_PATH + TRAIN_PKL, 'rb'))

    # Split train, vali set
    train_set, vali_set = train_test_split(df,
                                           test_size=0.2,
                                           random_state=RANDOM_STATE)

    # Drop columns without images
    train_set.dropna(subset=['image'], inplace=True)
    vali_set.dropna(subset=['image'], inplace=True)

    # Combine image name with path:
    imgs_train = train_set['image'].apply(
        lambda file: TRAINIMG_PATH + file + '.jpg'
        )
    imgs_vali = vali_set['image'].apply(
        lambda file: VALIIMG_PATH + file + '.jpg'
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

    # CNN Model
    cnn = KerasModelTrainner(model=av02_inceptv3(),
                             model_name='av02_InceptionV3_0601',
                             train_loader=train_loader,
                             vali_loader=vali_loader)

    cnn.compile(loss='mean_squared_error',
                optimizer='rmsprop'
                )

    cnn.fit(batch_size=128,
            epoch=1,
            augment=False,
            queue=10,
            log='av02_InceptionV3_0601.log')

    cnn.write_info(batch_size=128, epoch=1)
    cnn.save()


if __name__ == '__main__':
    train_nn()
    train_cnn()
