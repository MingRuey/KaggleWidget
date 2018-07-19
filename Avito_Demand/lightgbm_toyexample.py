import numpy
import pandas
import lightgbm
import matplotlib.pyplot as plt

train = pandas.DataFrame({'f1': [i for i in range(10)],
                          'f2': numpy.random.rand(10) - 0.5,
                          'f3': numpy.random.randint(10,20,size=10),
                          'f4': numpy.random.randint(2, size=10),
                          'f5': (numpy.random.rand(10)-0.5) * 50,
                          'label': (numpy.random.rand(10)-0.5) * 5
                          })

vali = pandas.DataFrame({'f1': [i for i in range(3)],
                         'f2': numpy.random.rand(3) - 0.5,
                         'f3': numpy.random.randint(10,20,size=3),
                         'f4': [1, 0, 1],
                         'f5': (numpy.random.rand(3)-0.5) * 50,
                         'label': (numpy.random.rand(3)-0.5) * 5
                        })

print('train: ')
print(train)
print('vali: ')
print(vali)

# create train/vali data
lgb_train = lightgbm.Dataset(data = train.drop('label', axis=1),
                             label = train['label']
                             )

lgb_vali = lightgbm.Dataset(data = vali.drop('label', axis=1),
                            label = vali['label'],
                            reference = lgb_train
                            )

# train parameters
params = {'device': 'cpu',
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mse',
          'learning_rate': 0.005,
          'num_leaves': 5,
          'max_depth': 3,
          'min_data_in_leaf': 1,
          }

# start training
eval_results = {}
gbm = lightgbm.train(params = params,
                     num_boost_round = 10,
                     train_set = lgb_train,
                     valid_sets = lgb_vali,
                     evals_result=eval_results,
                     verbose_eval=1
                     )

print('Plot metrics during training...')
ax = lightgbm.plot_metric(eval_results, metric='l2')
plt.show()

print('Plot feature importances...')
ax = lightgbm.plot_importance(gbm, max_num_features=5)
plt.show()

# #
# y = gbm.predict(vali.drop(['item_id', 'description', 'deal_probability'], axis=1).values)
# print('gbm: Feature importances:', list(gbm.feature_importance()))
# print('gbm: RMSE:', mean_squared_error(vali['deal_probability'], y) ** 0.5)
#
# test with txt
# gbm.save_model('model.txt')
# model_txt = lightgbm.Booster(model_file='model.txt')

#
# y = model_txt.predict(vali.drop(['item_id', 'description', 'deal_probability'], axis=1).values)
# print('model-txt: Feature importances:', list(model_txt.feature_importance()))
# print('model-txt: RMSE:', mean_squared_error(vali['deal_probability'], y) ** 0.5)
#
#
# test with pickle
# with open('model.pickle', 'wb') as fout:
#     pickle.dump(gbm, fout)
# with open('model.pickle', 'rb') as fin:
#     model_pkl = pickle.load(fin)
#
# y = model_pkl.predict(vali.drop(['item_id', 'description', 'deal_probability'], axis=1).values)
# print('model-pkl: Feature importances:', list(model_pkl.feature_importance()))
# print('model-pkl: RMSE:', mean_squared_error(vali['deal_probability'], y) ** 0.5)