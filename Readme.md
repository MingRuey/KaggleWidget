Customed and frequently used tools in Kaggle competitions.
===

> Codes are organized by their compititions.

>  TF_Utils: at draft stage, general support functions for building TF models  
>> store all data and label into tfrecord files, and use Dataset and Estimator API for trainning

>  9/17 update -- add TF_Utils   
>> Deprecation: CNN utils,  move old into Google_OpenImg  
>> Deprecation: KerasUtils, keras already merged with tensorflow, moved to Legacy               

About TF_Utils:
---

Image Data Pipeline:

    -- Universal data pipeline for image data and CNN, Usage:

    1. Define problem as Classification(Cls) or Object Detection(Oid), import build_cls_feature or build_oid_feature accordingly from img_feature_proto.py.
    2. Use above functions, write a class that wraps raw image to support .to_tfexample() method, which return a single tf.train.Example.
       (This API is defined in by ImgObjAbstract in WriteTFRecord.py)
    3. Create a generator for processing all images.
    4. Use write_tfrecord in WriteTFRecord.py to write TFRecord files.
    

Models:

    Some CNN models, currently including:
    
    UNet: full Unet implementation.
    Resnet.py: Resnet V2 blocks, bottleneck or not, can be used to create Resnet structure.
    FasterRCNN.py: Including Regional proposal networks(RPN) and final ROI layers for faster-RCNN. 
    InceptionResnetV2.py: Direct import of InceptionResnetV2 from tf.keras.applications.
    
    CustomLoss.py: Focal loss, smooth L1 loss, ...that are EITHER supports tf.keras.backend or tf.losses.

Config:

    At draft stage, aim to support easy training scenario setting.

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