# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:39:15 2018
@author: MRChou

Download image from the big json file, provided in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""
# for module
import ijson
import urllib.request
# for main()
import os
import functools
from multiprocessing import Pool


def img_download(url, filename, path):
    if not os.path.isfile(path+'\\'+filename): # download with such many files, it is worthy to check.
        try:
            file = open(path+'\\'+filename, 'wb')
            img = urllib.request.urlopen(url).read()
            if img:
                file.write(img)
        except (OSError, IOError) as err:
            print(err, url)
    return None


def json_parse(file):
    imgids = []
    urls = []
    for prefix, event, value in ijson.parse(file):
        try:
            if prefix.lower().endswith('id'): # dev: test prefix before value is important!
                imgids.append(str(value)+'.jpg')
            elif value.startswith('https://contestimg.wish.com'): # this could raise AttributeError!
                urls.append(value)
        except AttributeError as err:# if value or prefix is not string
            pass
        
        if len(urls) and len(imgids):
            yield (urls.pop(), imgids.pop())
    
    if len(urls): # if there are not enough image IDs:
        for url in urls:
            filename = url.split('/')[-1] + \
                       ('.jpg' if not url.split('/')[-1].endswith(('.jpeg','.jpg')) else '')
            yield url, filename

def main():
    try:
        file = open('data_train.json')
        path = os.getcwd() + '\\data_train'
        if not os.path.exists(path):
            os.makedirs(path)
            
        # download images in multiproccesses
        func = functools.partial(img_download, path=path)
        task = json_parse(file)
        with Pool(processes=10) as p:
            p.starmap(func, task)
            p.close()            
            p.join()
            
    except IOError as err:
        print(err)
    finally:
        file.close()

if __name__=='__main__':
    main()