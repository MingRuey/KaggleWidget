"""
Created on Apr. 29. 2018.
@author: cchsia
@author: mrchou

define different model used in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

def model_vgg16():
    model = VGG16(weights='imagenet', include_top=False)
    return model

def model_inceptionV3():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # flatten vs avgPooling: https://github.com/keras-team/keras/issues/8470
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # model.add(Dropout(0.5))
    x = Dense(1024, activation='relu')(x)
    # model.add(Dropout(0.5))
    predictions = Dense(228, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # don't changed the weight of base_model.
    for layer in base_model.layers:
        layer.trainable = False

    return model

def main():

    model = model_inceptionV3()
    model.summary()

    # compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    # use 2 GPU
    #parallel_model = multi_gpu_model(model, gpus=2)
    #parallel_moidel.fit(train_dataset, train_label, epochs=3, batch_size=300, validation_split=0.1, shuffle=True)

if __name__=='__main__':
    main()