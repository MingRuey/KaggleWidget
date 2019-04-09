import pickle
import lightgbm
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

PATH = '/archive/Avito/models/'

with open(PATH + 'av08_lgb-gbdt_0627.pickle', 'rb') as f_model:
    gbm = pickle.load(f_model)

    print('Plot feature importances...')
    ax = lightgbm.plot_importance(gbm,
                                  max_num_features=50,
                                  importance_type='split')
    plt.show()

    ax = lightgbm.plot_importance(gbm,
                                  max_num_features=50,
                                  importance_type='gain')
    plt.show()


# f03_train = load_npz(os.path.join(PATH, 'F03_train.npz'))


