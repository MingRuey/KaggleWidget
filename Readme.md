Record scripts for Kaggle competitions.

===
> Codes are organized by their compititions.

>  2020 update -- separate basic tools in MLBOX. TS_Utils and TF_Utils are going to be decrepted.
>  Jan. 2018 update -- add TS_Utils
>  Sep. 2017 update -- add TF_Utils
>> Deprecation: CNN utils,  move old into Google_OpenImg
>> Deprecation: KerasUtils, keras already merged with tensorflow, moved to Legacy

Prostate cANcer graDe Assessment (PANDA) Challenge
---

competitions info: https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/evaluation

    Apr. 22. 2020: Initial progress.md


Google_Inclusive
---

competition info: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    sanity check on image labels:
    Inclusive_script_sanity-check.py

    pickle image labels:
    Inclusive_script_pickle_id-label-map.py

    Use TF_Utils.ImgPipeline structure:
    turning images into tfrecord files -- Inclusive_script_img_to_tfrecord.py


RSNA_Pneumonia
---

competition info: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    pickle image labels:
    RSNA_script_pickle_id-label-map.py

    Use TF_Utils.ImgPipeline structure:
    turning images into tfrecord files -- RSNA_script_img_to_tfrecord.py

    Models:
    for trainning faster-RCNN -- RSNA_train_frcnn.py
    for pre-classifying images -- RSNA_train_classifier.py, which uses InceptionResnetV2 from keras.

Google_OpenImg
---

competition info: https://www.kaggle.com/c/avito-demand-predict

    Using Google_OpenImg.CnnUtils for:
    pickle image labels -- GoogleImg_CreateLabelMap.py

    turning images into tfrecord files:
    GoogleImg_ToTFR.py

    make prediction by TF model:
    GoogleImg_scripts.py


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