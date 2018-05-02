"""
Created on May. 02. 2018
@author: MRChou

Generate report of a given model on some validation set,
precision, recall, F1 score, and performance on each label also.

"""

import numpy
from matplotlib import pyplot as plt
from FGVC5_iMfashion.iMfashion_ImgBatchLoader import ImgBatchLoader

# Eval_matrix is used to record the current accuracy of prediction,
# since predictions can not/should not make on the whole validation set at once,

# Accuracy is recorded by counting the total positives/true positives/false negatives via .update() method.
# Use label_report/report method to get the final statistics.
class eval_matrix(numpy.ndarray):

    def __new__(cls, size=(0,0), dtype='uint32'):
        return numpy.ndarray.__new__(cls, [], dtype=dtype)

    def __repr__(self):
        if not self.ndim:
            return "([], No data have been loaded.)"

        return 'Labels:    {0} '.format(self[0, :].tolist()) + '\n' + \
               'Positives: {0} '.format(self[1, :].tolist()) + '\n' + \
               'True pos : {0} '.format(self[2, :].tolist()) + '\n' + \
               'False neg: {0} '.format(self[3, :].tolist())

    # With the prediction values and labels.
    # Caculating the total positive, true positive and false negative.
    def update(self, predict, label):
        assert predict.shape == label.shape, 'eval_matrix.update(): Shapes of prediction and label do not match.'
        if not self.ndim:
            self.resize((4, label.shape[1]), refcheck=False)
            self[0, :] = range(1, label.shape[1]+1)

        p = predict >= 1
        l = label >= 1
        if p.ndim !=1:
            self[1, :] += p.sum(axis=0, dtype='uint32')  # total positive
            self[2, :] += numpy.logical_and(p, l).sum(axis=0, dtype='uint32') # true positive
            self[3, :] += numpy.logical_and(numpy.logical_not(p), l).sum(axis=0, dtype='uint32')  # false negative

        else: # update with only one data (i.e. only one row).
            self[1, :] += p  # total positive
            self[2, :] += numpy.logical_and(p, l) # true positive
            self[3, :] += numpy.logical_and(numpy.logical_not(p), l) # false negative

    # Display the precision, recall of each label in bar chart.
    def label_report(self):
        fig, ax = plt.subplots()
        width = 0.05

        # precision
        p = ax.bar(self[0, :]-width/2, self[2, :]/self[1, :], width, color='g', label='precision')
        # recall
        r = ax.bar(self[0, :]+width/2, self[2, :]/(self[3, :]+self[2, :]), width, color='r', label='recall')

        ax.set_xlabel('Labels')
        ax.set_ylabel('Scores')
        ax.set_title('Scores by Labels')
        ax.set_xticks(self[0, :])
        ax.set_xticklabels((self[0, :]))
        ax.legend()
        fig.tight_layout()
        plt.show()

    # Return the precision, recall and f1 score overvall
    def report(self):
        p = self[2, :].sum() / self[1, :].sum()                 # precision over all labels
        r = self[2, :].sum() / (self[3, :].sum()+self[2,:].sum()) # recall over all labels
        f1 = 0 if p+r == 0 else 2*p*r/(p+r)                    # f1 over all labels
        return "precision:{0:4.3f}, recall:{1:4.3f}  with F1 score:{2:4.3f}".format(p,r,f1)

def validate(model, vali_path, label, batchsize=4096):

    eval = eval_matrix()
    batchs = ImgBatchLoader(img_path=vali_path, img_label=label, img_size=model.input_shape[1:])

    for img, label in batchs.generator(batch_size=batchsize, epoch=1, shuffle=False):
        eval.update(predict=model.predict(img), label=label)

    return eval

# ---
# Test the module
import pickle
import imageio
def main():
    path = '/home/mrchou/code/KaggleWidget/'

    # generate test data.
    def test_data():
        # create test images and pickle file
        label = numpy.zeros((10, 15))

        for i in range(10):
            imageio.imwrite(path + str(i) + '.jpg', 10 * (i + 1) * numpy.ones((300, 300, 3), dtype='uint8'))
            label[i, 0] = i
            label[i, i + 1] = 1

        fw = open(path + 'labels.pickle', 'wb')
        pickle.dump(label, fw)
        fw.close()

    test_data()

    class test_model():
        def __init__(self, batchsize=None):
            self.input_shape = (batchsize, 300, 300, 3)
            self.__batchsize = batchsize

        def predict(self, img):
            return numpy.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]]*self.__batchsize)

    label = 'labels.pickle'
    batchsize=5
    model = test_model(batchsize=batchsize)
    matrix = validate(model=model, vali_path=path, label=label, batchsize=batchsize)

    print(repr(matrix))
    print(matrix.report())
    matrix.label_report()

if __name__=='__main__':
    main()



