import numpy
import pickle
from sklearn.tree import DecisionTreeRegressor
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


def eval_weight():
    """Compute the label report of given weight.."""

    pkls = ['IncepV3_0506_iM2.pickle',
            'Xception_iM3.pickle',
            'IncepV3+drop_0514_iM7.pickle',
            'IncepV3_0520_iM10.pickle',
            'DenseNet169_0524_iM12.pickle',
            'Resnet50_0524_iM13.pickle'
            ]

    label = pickle.load(open(vali_label, 'rb'))

    weights = [[0.4, 0.05, 0.05, 0.3, 0.05, 0.15], [0.35, 0.05, 0.05, 0.3, 0.05, 0.2]]
    threshold = 0.2

    with open('rule-based_statistic.txt', 'w') as f:
        for weight in weights:
            mat = numpy.zeros(label[:, 1:].shape)
            for i, pkl in enumerate(pkls):
                predict = pickle.load(open(ENSEMBLE_PATH+pkl, 'rb'))  # check if img id match
                if numpy.all(predict[:, 0] == label[:, 0]):
                    mat += predict[:, 1:]*weight[i]
                else:
                    raise ValueError('{0}: label id does not match'.format(pkl))

            eval_mat = eval_matrix()
            eval_mat.update((mat > threshold), label=label[:, 1:])
            f.write(''.join(str(i)+' ' for i in weight) + '\n')
            f.write('AP: ' + ''.join(str(i)+' ' for i in eval_mat[1, :]) + '\n')
            f.write('Tp: ' + ''.join(str(i)+' ' for i in eval_mat[2, :]) + '\n')
            f.write('Fn: ' + ''.join(str(i)+' ' for i in eval_mat[3, :]) + '\n')


def label_learner():
    """A soft-max regressor for compensate the label difference between vali set and predction."""

    pkls = ['IncepV3_0506_iM2.pickle',
            'Xception_iM3.pickle',
            'IncepV3+drop_0514_iM7.pickle',
            'IncepV3_0520_iM10.pickle',
            'DenseNet169_0524_iM12.pickle',
            'Resnet50_0524_iM13.pickle'
            ]

    label = pickle.load(open(vali_label, 'rb'))

    weight = [0.35, 0.05, 0.05, 0.3, 0.05, 0.2]

    mat = numpy.zeros(label[:, 1:].shape)
    for i, pkl in enumerate(pkls):
        predict = pickle.load(open(ENSEMBLE_PATH+pkl, 'rb'))  # check if img id match
        if numpy.all(predict[:, 0] == label[:, 0]):
            mat += predict[:, 1:]*weight[i]
        else:
            raise ValueError('{0}: label id does not match'.format(pkl))

    reg = DecisionTreeRegressor(max_depth=10, criterion='mse')
    reg.fit(mat, label[:, 1:])

    threshold = 0.99
    with open('TreeRegressor.csv', 'w') as f:
        f.write('image_id,label_id\n')
        for n, row in enumerate(reg.predict(mat)):
            imgid = int(predict[n, 0])
            labels = numpy.where(row > threshold)[0] + 1  # note that index+1 = actual labels
            f.write(str(int(imgid)) + ',' +
                    ''.join(str(label) + ' ' for label in labels) +
                    '\n'
                    )

    threshold = 0.01
    with open('TreeRegressor_negtive.csv', 'w') as f:
        f.write('image_id,label_id\n')
        for n, row in enumerate(reg.predict(mat)):
            imgid = int(predict[n, 0])
            labels = numpy.where(
                                numpy.logical_and(row < threshold, mat[n, :] > 0.2)
                                )[0] + 1  # note that ndex+1 = actual labels
            f.write(str(int(imgid)) + ',' +
                    ''.join(str(label) + ' ' for label in labels) +
                    '\n'
                    )


label_learner()