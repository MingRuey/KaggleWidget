from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# base_model, can use VGG, ResNet or others
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

# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)


