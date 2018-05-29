import os
import cv2
import numpy
import pickle
import itertools
from keras.models import load_model
from iMfashion_ValidateModel import eval_matrix


MODEL_PATH = '/archive/iMfashion/models/'
ENSEMBLE_PATH = '/archive/iMfashion/ensemble/'

vali_path = '/rawdata/FGVC5_iMfashion/imgs_vali/'
vali_label = '/archive/iMfashion/labels/labels_validation.pickle'

# Raw models other than IncepV3
model3 = 'Xception_iM3.h5'  # iM3  F1 0.544
model12 = 'DenseNet169_0524_iM12.h5'  # iM12 F1 0.545
model13 = 'Resnet50_0524_iM13.h5'  # iM13 F1 0.560

# IncepV3 and its variant
model2 = 'IncepV3_0506_iM2.h5'  # iM2  F1 0.582
model7 = 'IncepV3+drop_0514_iM7.h5'  # iM7  F1 0.539, IncepV3+dropout
model10 = 'IncepV3_0520_iM10.h5'  # iM10 F1 0.573, Grabcut


def generate_predict():
    """Store the prediction of model on a image set into a pickle file"""
    h5s = [model2, model3, model7, model12, model13]

    for h5 in h5s:
        model = load_model(MODEL_PATH + h5)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')

        files = [f for f in os.listdir(vali_path) if f.lower().endswith('.jpg')]
        matrix = numpy.zeros((len(files), 229))  # 1 img id column + 228 labels prediction

        for file in files:
            img = cv2.imread(os.path.join(vali_path, file))[..., ::-1]  # BGR by default, convert it into RGB
            img = cv2.resize(img, (300, 300))

            imgid = int(file[:-4])
            matrix[imgid-1, 0] = imgid
            matrix[imgid-1, 1:] = model.predict(img[numpy.newaxis, :])

        matrix = matrix[matrix[:, 0].argsort()]  # sort by image id
        with open(h5[:-3]+'_test.pickle', 'wb') as f:
            pickle.dump(matrix, f)


def search_weight():
    """Grid search the weight of ensembling models."""

    f_out = 'search_weights_for_iM2-3-7-10-12-13.txt'
    header = 'F1 of: [iM2, 3, 7, 10, 12, 13] are: [0.582, 0.544, 0.539, 0.573, 0.545, 0.560] \n'

    pkls = ['IncepV3_0506_iM2.pickle',
            'Xception_iM3.pickle',
            'IncepV3+drop_0514_iM7.pickle',
            'IncepV3_0520_iM10.pickle',
            'DenseNet169_0524_iM12.pickle',
            'Resnet50_0524_iM13.pickle'
            ]

    label = pickle.load(open(vali_label, 'rb'))

    weights = [weight for weight in
               itertools.product([i/20 for i in range(1, 22-len(pkls))], repeat=len(pkls))
               if sum(weight) == 1
               ]
    with open(f_out, 'w') as f:
        f.write(header)
        for weight in weights:
            for threshold in [0.08 + 0.03*i for i in range(11)]:
                mat = numpy.zeros(label[:, 1:].shape)
                for i, pkl in enumerate(pkls):
                    predict = pickle.load(open(ENSEMBLE_PATH+pkl, 'rb'))  # check if img id match
                    if numpy.all(predict[:, 0] == label[:, 0]):
                        mat += predict[:, 1:]*weight[i]
                    else:
                        raise ValueError('{0}: label id does not match'.format(pkl))

                eval_mat = eval_matrix()
                eval_mat.update((mat > threshold), label=label[:, 1:])
                f.write('weight {0}; thres {1}; (p, r, f1) is {2:5.4f}; {3:5.4f}; {4:5.4f} \n'.format(
                        weight, threshold, *eval_mat.get_f1()
                        ))


def ensemble_submission():
    """Create submission with ensemble of given models and wieght"""

    csv_name = 'submission_iM2-3-7-10-12-13.csv'

    pkls = ['IncepV3_0506_iM2_test.pickle',
            'Xception_iM3_test.pickle',
            'IncepV3+drop_0514_iM7_test.pickle',
            'IncepV3_0520_iM10_test.pickle',
            'DenseNet169_0524_iM12_test.pickle',
            'Resnet50_0524_iM13_test.pickle'
            ]

    weight = [0.35, 0.05, 0.05, 0.3, 0.05, 0.2]
    threshold = 0.2

    with open(csv_name, 'w') as f:
        f.write('image_id,label_id\n')

        for i, pkl in enumerate(pkls):
            predict = pickle.load(open(ENSEMBLE_PATH+pkl, 'rb'))  # check if img id match

            if not('mat' in locals()):
                mat = numpy.zeros(predict[:, 1:].shape)
            mat += predict[:, 1:] * weight[i]

        for n, row in enumerate(mat):
            imgid = int(predict[n, 0])
            labels = numpy.where(row > threshold)[0] + 1  # note that index+1 = actual labels
            f.write(str(int(imgid)) + ',' +
                    ''.join(str(label) + ' ' for label in labels) +
                    '\n'
                    )


if __name__ == '__main__':
    search_weight()