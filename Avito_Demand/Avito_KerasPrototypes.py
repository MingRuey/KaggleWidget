from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3


def av01_nn():
    """A simple two 2048x2048 FC model"""
    model = Sequential()
    model.add(Dense(2048, input_dim=1611, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def av02_inceptv3():
    """Inception V3 with custom top"""
    base_model = InceptionV3(input_shape=(300, 300, 3),
                             weights='imagenet',
                             include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

