Customed and frequently used tools in Kaggle competitions.
===

>Codes are organized by their compititions.

>Additionally, KerasUtils.py stores some useful modules/patterns for building models from Keras.


Avito Demand
---

competition info: https://www.kaggle.com/c/avito-demand-prediction/

    Modules for data cleaning and various enconding:
    Avtio_CsvProcess.py

    For generating image features:
    Avito_ImageFeature.py

    For generating image features:
    Avito_KerasPrototypes, LgbPrototype, XgbPrototype, FfmPrototype ...

    Scripts for trainning models:
    Avito_TrainScripts.py

    For creating submittion file:
    Avito_modelsubmission.py

FGVC5 iMfashion
---
competition info: https://www.kaggle.com/c/avito-demand-prediction/

Across the entire competition, we stick to simple raw data pipeline:

image -- We download all images from http links, stored on a local server.
         And load images via a generator in iMfashion_ImgBatchLoader for Keras utils like fit_generator, ... etc.

label -- We extract the labels from Json file given, stored them in a numpy array with one-hot like format.
         For deatiled format, check out: iMfashion_JsonExtractLabel.


    Extracting data from the Json file:
    iMfashion_JsonExtractLabels.py, iMfashion_JsonGetImg.py, iMfashion_CheckBrokenImgs.py

    Selecting a smaller subset of data by check similarity betweem labels:
    iMfashion_TrainSetFilter.py

    Attempts on image masking(mainly removing image background)
    iMfashion_ObjectDetection.py

    Generator for loading images into Keras models
    iMfashion_ImgBatchLoader.py

    Models(all using Keras)
    iMfashion_kerasPrototype.py

    For creating submission file and examine the model performance:
    iMfashion_ValidateModel.py

    For ensembling on the submission files
    iMfashion_ModelEnsemble.py

