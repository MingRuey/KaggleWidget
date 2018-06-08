import xlearn

PATH = '/archive/Avito/data_preprocess/'

# Train model:
ffm_model = xlearn.create_ffm()
ffm_model.setTrain(PATH + 'FFM_train_set.txt')
ffm_model.setValidate(PATH + 'FFM_vali_set.txt')
ffm_model.disableNorm()

param = {'k': 10,
         'task': 'reg',
         'opt': 'adagrad',
         'lr': 0.1,
         'lambda': 0.002,
         'metric': 'rmse',
         'epoch': 100}

ffm_model.fit(param, './av03_ffm_0607.txt')
ffm_model.show()