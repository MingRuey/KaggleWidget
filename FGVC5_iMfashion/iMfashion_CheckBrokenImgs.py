# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 23:52:37 2018
@author: MRChou

Compare a folder holding images with a json file from Kaggle iMaterialist Fashion.
Download missing and broken images, if any.
"""

import os

from iMfashion_JsonExtractLabel import urlfromjson
from iMfashion_JsonGetImg import img_download

# compare imgs in a given path with a image set,
# return the difference
def missing_imgs(imgs_path, imgs):
    imgs_in_target = set(i for i in os.listdir(imgs_path) if i.endswith('.jpg'))
    imgs = set(imgs)
    return imgs - imgs_in_target

# return the empty imgs in a given path
def empty_imgs(imgs_path):
    empty_imgs = set(i for i in os.listdir(imgs_path) \
                            if i.endswith('.jpg') and \
                               os.stat(os.path.join(imgs_path,i)).st_size == 0)
    return empty_imgs
    
def main():
    file = open('data_train.json')
    urls = urlfromjson(file)
    target_path = 'Y:/FGVC5_iMfashion/imgs_train'
    output_path = 'Y:/FGVC5_iMfashion/imgs_broken_train'
    
    imgs_miss = missing_imgs(target_path ,{str(i)+'.jpg' for i in urls.keys()})
    if imgs_miss:
        print('{0} imgs are missing from the target folder.'.format(len(imgs_miss)))
        print('Start downloading missing imgs...')
        for img in imgs_miss:
            imgid = int(os.path.splitext(img)[0])
            img_download(urls[imgid], img, output_path)
    print('Finish checking missing imgs.')
    
    imgs_empty = empty_imgs(target_path)
    fail = 0
    if imgs_empty:
        print('{0} imgs are empty from the target folder.'.format(len(imgs_empty)))
        print('Start removing and re-downloading ...')
        for img in imgs_empty:
            imgid = int(os.path.splitext(img)[0])
            if not img_download(urls[imgid], img, output_path):
                fail +=1
    print('Finish checking empty imgs, with {0} remaining empty'.format(fail))
    
if __name__=='__main__':
    main()