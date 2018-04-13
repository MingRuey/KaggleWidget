# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:39:15 2018
@author: MRChou

Download image from the big json file, provided in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""

import ijson
import os
import urllib

def img_download(url, file):
    try:
        img = urllib.request.urlopen(url).read()
        if img:
            file.write(img)
    except (OSError, IOError) as err:
        print(err)
    return None

def json_parse(file, path_img_dl=None):
    path = os.getcwd() + '\\' + ('images' if not path_img_dl else path_img_dl)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # parsing jason file
    print('...start parsing')
    urls = []
    imgId = [] # two big lists get the shit done, while writing in coroutine way may be better.
    for prefix, event, value in ijson.parse(file):
        try:
            if prefix.lower().endswith('id'): # dev: test prefix before value is important!
                imgId.append(value)
            elif value.startswith('https'): # this could raise AttributeError!
                urls.append(value)
        except AttributeError as err: # if value or prefix is not string
            pass
    print('...end parsing, start dowloading')
    
    # downloading images
    for url in urls:
        try:
            filename = str(imgId.pop(0)) + '.jpg'
        except IndexError: # there are not enough image IDs:
            filename = url.split('/')[-1] + \
                       ('.jpg' if not url.split('/')[-1].endswith(('.jpeg','.jpg')) else '')
        try:          
            file = open(path+'\\'+filename, 'wb')
            img_download(url, file)
        except IOError as err:
            print(err)
        finally:
            file.close()
    print('...finished')
    return None

def main():
    try:
        file = open('data_test.json')
        json_parse(file, path_img_dl='data_test')
    except IOError as err:
        print(err)
    finally:
        file.close()

if __name__=='__main__':
    main()