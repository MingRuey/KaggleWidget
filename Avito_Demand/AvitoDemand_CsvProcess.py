# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:32:19 2018
@author: MRChou

Code used for preprocessing Csv data, including tools for processing large csv files. 
Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""

import pandas

PATH = '/archive/Avito/russian_translation/'
def russian_to_en(df_data):
    """Inplace translation for a dataframe from Russian to English"""
    tranlate_file = {'region':  'russian_region_names_in_english.csv',
                     'city': 'russian_city_names_in_english.csv',
                     'parent_category_name': 'parent_product_categories.csv',
                     'category_name': 'product_categories.csv',
                     'param_1': 'param_1.csv',
                     'param_2': 'param_2.csv',
                     'param_3': 'param_3.csv'}
    
    # Iterate over input dataframe columns,
    # Use files from: www.kaggle.com/kaparna/translations/data, if possible.
    # For 'description' column, use translate module.
    for col in df_data.columns:
        file = tranlate_file.get(col, None) 
        if file: 
            # build up the mapping from Russian to English.
            file = pandas.read_csv(PATH+file)
            file.columns = ['rus', 'en']
            convert = {row['rus']: row['en'] for index, row in file.iterrows()}
            
            # translate column into English
            df_data[col] = df_data[col].map(convert, na_action = 'ignore')

def main():
    files = ['train.csv', 'train_active.csv',
             'test.csv', 'test_active.csv',
             'periods_train.csv', 'periods_test.csv']

    file_path = '/rawdata/Avito_Demand/'
    output_path = '/archive/Avito/'
    for file in files:
        df_data = pandas.read_csv(file_path+file)
        russian_to_en(df_data)
        df_data.to_pickle(output_path+file[:-4]+'.pickle')

if __name__=='__main__':
    main()