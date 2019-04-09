from iMfashion_kerasPrototype import model_DenseNet169, model_Resnet50
from iMfashion_kerasPrototype import model_trainner, model_continue


def main():

    train_path = '/rawdata/FGVC5_iMfashion/imgs_train/'
    train_label = '/archive/iMfashion/labels/labels_train.pickle'
    vali_path = '/rawdata/FGVC5_iMfashion/imgs_validation/'
    vali_label = '/archive/iMfashion/labels/labels_validation.pickle'

    model = model_continue('/archive/iMfashion/models/Resnet50_0524_iM13.h5')

    s = model_trainner(model=model,
                       model_name='Resnet50_0526_iM14',
                       train_path=train_path,
                       train_label=train_label,
                       vali_path=vali_path,
                       vali_label=vali_label,
                       )

    s.fit(optimizer='rmsprop',
          loss='binary_crossentropy',
          batch_size=64,
          epoch=5,
          augmenting=False,
          multi_gpu=2,
          log=True
          )

if __name__=='__main__':
    main()
