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
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

import logging
logging.basicConfig(format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filename='training.log', level=logging.INFO
        )
import matplotlib.pyplot as plt
#from FGVC5_iMfashion.iMfashion_ImgBatchLoader import ImgBatchLoader
from iMfashion_ImgBatchLoader import ImgBatchLoader

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
    # for layer in base_model.layers:
    #     layer.trainable = False
    
    return model
    

def plot_history(fit_history):
    # list all data in history
    # print(fit_history.history.keys())
    # summarize history for loss
    # plt.plot(fit_history.history['loss'])
    # plt.plot(fit_history.history['val_loss'])
    plt.plot(fit_history.epoch_losses)
    plt.plot(fit_history.epoch_val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('lost.png')

class train_history(Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.epoch_val_losses = []
        self.epoch_losses = []
    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_val_losses.append(logs.get('val_loss'))
        self.epoch_losses.append(logs.get('loss'))

def main():
    model = model_inceptionV3()
    model.summary()

    train_path = '/rawdata/FGVC5_iMfashion/imgs_train/'
    train_label = '/archive/iMfashion/labels/labels_train.pickle'
    vali_path = '/rawdata/FGVC5_iMfashion/imgs_validation/'
    vali_label = '/archive/iMfashion/labels/labels_validation.pickle'
    
    train_loader = ImgBatchLoader(img_path=train_path, img_label=train_label)
    vali_loader = ImgBatchLoader(img_path=vali_path, img_label=vali_label)
    '''
    for i in train_loader.generator(10, shuffle=True):
        print(i[0].shape)
        print(i[1].shape)
        print('=====')
    '''
    # use 2 GPU
    parallel_model = multi_gpu_model(model, gpus=2)
    
    # compile the model
    
    # parallel_moidel.fit(train_dataset, train_label, epochs=3, batch_size=300, 
    #         validation_split=0.1, shuffle=True)
    
    parallel_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    parallel_model.summary()

    # checkpoint
    '''
    filepath="best_w.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
            verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    '''
    
    history = train_history()
    parallel_model.fit_generator(generator=train_loader.generator(128),
            validation_data=vali_loader.generator(128),
            validation_steps=9900//128,
            steps_per_epoch=1014547//128, epochs=5,
            use_multiprocessing=True,
            workers=16,
            max_queue_size=10,
            callbacks=[history]#, checkpoint]
            )
    '''
    parallel_model.fit_generator(generator=vali_loader.generator(128),
            validation_data=vali_loader.generator(128),
            validation_steps=9900//128,
            steps_per_epoch=9900//128, epochs=2,
            use_multiprocessing=True,
            workers=16,
            max_queue_size=10,
            callbacks=[history]#, checkpoint]
            )
    '''
    # score = parallel_model.evaluate_generator(vali_loader.generator(128), 990/128, workers=16)
    # scores = parallel_model.predict_generator(vali_loader.generator(128), 990//128, workers=16)
    # print(score)
    # check point will save the best model (val_loss lowest)
    model.save('test_model_all_vali_0503_nofit.h5')
    
    logging.info(history.batch_losses)
    logging.info(history.epoch_losses)
    logging.info(history.epoch_val_losses)
    
    # plot training process
    # plot_history(history)

if __name__=='__main__':
    main()
