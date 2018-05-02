# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:39:15 2018
@author: MRChou

Download images from the big json file, provided in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""
# for module
import ijson
import urllib.request
from iMfashion_JsonExtractLabel import urlfromjson
# for main()
import os
import functools
from multiprocessing import Pool

def img_download(url, filename, path):
    if not os.path.isfile(path+'\\'+filename): # if the file already exists, pass it.
        try:
            img = urllib.request.urlopen(url)
            img = img.read()
            if img:
                file = open(path+'\\'+filename, 'wb')
                file.write(img)
                file.close()
        except (OSError, IOError) as err:
            print('-- while downloadning {0}, {1}'.format(filename, err))
            return None
    return True

def urlgenerator_json(file):
    urls = urlfromjson(file)
    for imgid in urls:
        yield urls[imgid], str(imgid)+'.jpg'
    
def urlgenerator_ijson(file):
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
        path = 'Y:/FGVC5_iMfashion/imgs_train'
        if not os.path.exists(path):
            os.makedirs(path)
            
        # download images in multiproccesses
        func = functools.partial(img_download, path=path)
        task = urlgenerator_json(file)
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