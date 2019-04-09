
# --- display multiple image in matplotlib ---
from matplotlib import pyplot as plt
from keras.models import load_model
from iMfashion_ImgBatchLoader import ImgBatchLoader
from iMfashion_ValidateModel import validate, submission
from iMfashion_kerasPrototype import model_IncepV3_withDrop, model_trainner

train_path = '/rawdata/FGVC5_iMfashion/imgs_train/'
train_label = '/archive/iMfashion/labels/labels_train.pickle'
vali_path = '/rawdata/FGVC5_iMfashion/imgs_validation/'
vali_label = '/archive/iMfashion/labels/labels_validation.pickle'

def multidisplay(img1, img2):
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    plt.show()


def vali_on_valiset():
    model = load_model('/archive/iMfashion/models/Resnet50_0524_iM13.h5')

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.summary()

    imgs_vali = ImgBatchLoader(img_path=vali_path, img_label=vali_label)
    imgs_vali = imgs_vali.generator(batch_size=1, epoch=1, shuffle=False)
    print(model.evaluate_generator(imgs_vali, steps=9897 ,max_queue_size=10, workers=16, use_multiprocessing=True))

    for i in [0.1, 0.2, 0.3]:
        s = validate(model, vali_path=vali_path, label=vali_label, threshold=i, batch_size=3299)
        print(s.report())

def submit_on_testset():

    model = load_model('/archive/iMfashion/models/Resnet50_0524_iM13.h5')

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.summary()

    test_path = '/rawdata/FGVC5_iMfashion/imgs_test'
    submission(model, 'Resnet50_iM13.csv', test_path=test_path, img_size=(300, 300, 3), threshold=0.2)
