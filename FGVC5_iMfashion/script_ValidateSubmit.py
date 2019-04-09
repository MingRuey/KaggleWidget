import os
from keras.models import load_model
from iMfashion_ImgBatchLoader import ImgBatchLoader
from iMfashion_ValidateModel import validate, submission

train_path = '/archive/iMfashion/preprocess/imgs_train/'
train_label = '/archive/iMfashion/labels/labels_train.pickle'
vali_path = '/archive/iMfashion/preprocess/imgs_validation/'
vali_label = '/archive/iMfashion/labels/labels_validation.pickle'
test_path = '/archive/iMfashion/preprocess/imgs_test/'

models = ['/archive/iMfashion/models/IncepV3_0520_iM10.h5']

f = open('evaluate.log', 'w')
for h5 in models:
    model = load_model(h5)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    f.write(h5)

    imgs_vali = ImgBatchLoader(img_path=vali_path, img_label=vali_label)
    imgs_vali = imgs_vali.generator(batch_size=1, epoch=1, shuffle=False)

    s = validate(model, vali_path=vali_path, label=vali_label, threshold=0.3, batch_size=3299)
    f.write('Threshold : 0.3 \n')
    f.write(s.report())
    s = validate(model, vali_path=vali_path, label=vali_label, threshold=0.2, batch_size=3299)
    f.write('Threshold : 0.2 \n')
    f.write(s.report())
    s = validate(model, vali_path=vali_path, label=vali_label, threshold=0.1, batch_size=3299)
    f.write('Threshold : 0.1 \n')
    f.write(s.report())

    model.evaluate_generator(imgs_vali, steps=9897, max_queue_size=10, workers=16, use_multiprocessing=True)

    submission(model, os.path.basename(h5)+'.csv', test_path=test_path)
