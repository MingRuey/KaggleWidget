# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:39:15 2018
@author: MRChou

Download images from the big json file, provided in iMaterialist Challenge(Fashion):
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018

"""
# for module
import os
import functools
import concurrent.futures as futures

import urllib.request
import ijson

from iMfashion_JsonExtractLabel import urlfromjson


def img_download(task, path):
    """Download the image from the *url, store it as *filename inside the *path."""
    url, filename = task
    if not os.path.isfile(path+'/'+filename):  # if the file already exists, pass it.
        try:
            img = urllib.request.urlopen(url)
            img = img.read()
            if img:
                file = open(path+'/'+filename, 'wb')
                file.write(img)
                file.close()
        except (OSError, IOError) as err:
            print('-- while downloadning {0}, {1}'.format(filename, err))
            return None
    return True


def urlgenerator_json(file):
    """Create a generator of urls from a json file."""
    urls = urlfromjson(file)
    for imgid in urls:
        yield urls[imgid], str(imgid)+'.jpg'


def urlgenerator_ijson(file):
    """Create a generator of urls from a json file, w/o loading entire json."""
    imgids = []
    urls = []
    for prefix, event, value in ijson.parse(file):
        try:
            if prefix.lower().endswith('id'):  # dev: test prefix before value is important!
                imgids.append(str(value)+'.jpg')
            elif value.startswith('https://contestimg.wish.com'):  # this could raise AttributeError!
                urls.append(value)
        except AttributeError:  # if value or prefix is not string
            pass
        
        if len(urls) and len(imgids):
            yield (urls.pop(), imgids.pop())
    
    if len(urls):  # if there are not enough image IDs:
        for url in urls:
            filename = url.split('/')[-1] + \
                       ('.jpg' if not url.split('/')[-1].endswith(('.jpeg', '.jpg')) else '')
            yield url, filename


def main():

    with open('/rawdata/FGVC5_iMfashion/data_test.json') as file:
        path = '/rawdata/FGVC5_iMfashion/imgs_test'
        if not os.path.exists(path):
            os.makedirs(path)
            
        # download images in multiproccesses
        func = functools.partial(img_download, path=path)
        task = urlgenerator_ijson(file)
        with futures.ThreadPoolExecutor(16) as exe:
            res = exe.map(func, task)


if __name__ == '__main__':
    main()
