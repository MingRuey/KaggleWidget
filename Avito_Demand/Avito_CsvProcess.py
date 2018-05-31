# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:32:19 2018
@author: MRChou

Code used for preprocessing Csv data.
Used for the Avito Demand Prediction Challenge:
https://www.kaggle.com/c/avito-demand-prediction/

"""
import numpy
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
            df_data[col] = df_data[col].map(convert, na_action='ignore')


# All followings are direct copy-paste from: Avito_FeatureEngineer.ipynb
# Check them out on server.
def city_rename(df, target=None):
    """Inpace add prefix and replace white space with underscore."""
    if not target:
        target = ['city']
    for col in target:
        df[col] = df[col].apply(
            lambda text: col + '-' + str(text).replace(' ', '_'))
    return None


def fillna_negtive1(df, target=None):
    """Inplace replace NaN with -1"""
    if not target:
        target = ['price', 'image_top_1']
    for col in target:
        df[col] = df[col].fillna(-1)
    return None


def encoding_labelcount(df, target=None):
    """Inpalce map label into the global cumultive count(Label Count)"""
    if not target:
        target = ['user_id', 'title']

    norm = round(
        df.shape[0] / 10000)  # normalize the count by /per 100000 entries
    for col in target:
        df[col + '_labelcount'] = df[col].map(df[col].value_counts()) / norm
        df.drop([col], axis=1, inplace=True)
    return None


def encoding_median(df, target=None):
    """Inplace map label into global median(i.e. mean-encoding with median)"""
    # It has been checked that the followings has enough data on each category
    # This is done via: www.csvexplorer.com)
    if not target:
        target = ['region', 'parent_category_name', 'category_name']
    for col in target:
        median = df.groupby(col).deal_probability.median()
        df[col + '_target_mean'] = df[col].map(median)
        df.drop([col], axis=1, inplace=True)
    return None


def encoding_onehot(df, target=None):
    """Add columns of the one-hot encoding, and drop the original ones."""
    if not target:
        target = ['user_type', 'city']
    for col in target:
        # Following is exactily the df.join() but is inplace.
        one_hot = pandas.get_dummies(df[col])
        for item in one_hot:
            df[item] = one_hot[item]
        df.drop([col], axis=1, inplace=True)
    return None


def encoding_date(df, target=None):
    """Inpalce add 2 columns: weekday and month, and drop the original date"""
    if not target:
        target = ['activation_date']
    for col in target:
        df[col] = pandas.to_datetime(df[col], format='%Y-%m-%d')
        df[col + '_month'] = df[col].dt.month
        df[col + '_weekday'] = df[col].dt.dayofweek
        df.drop([col], axis=1, inplace=True)
    return None


def encoding_nan_binarize(df, target=None):
    """Inplace Binarize column with NaN => 0, non-Nan =>1"""
    if not target:
        target = ['param_1', 'param_2', 'param_3', 'image']
    for col in target:
        df[col + '_or_not'] = numpy.where(df[col].isnull(), 0, 1)
        df.drop([col], axis=1, inplace=True)
    return None


def encoding_coltoback(df, target=None):
    """Move these two columns to the back of data"""
    if not target:
        target = ['item_seq_number', 'image_top_1', 'description',
                  'deal_probability']
    col_names = list(df)
    for col in target:
        col_names.append(col_names.pop(col_names.index(col)))
    df = df.loc[:, col_names]
    return df


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


if __name__ == '__main__':
    main()
