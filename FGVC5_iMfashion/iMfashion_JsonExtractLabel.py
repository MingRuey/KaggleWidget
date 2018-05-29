# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:22:54 2018
@author: MRChou

To extract the labels(called annotations in json file) of iMaterial Challenge --Fashion.

Reads json file, stores annotations in a numpy array and pickles the array.
"""

import json
import numpy
import pickle


def urlfromjson(file):
    """Extract URLs from a json file in iMfashion competition"""
    try:
        urls = json.load(file)
        urls = urls['images'] 
        # urls is a list: 
        # [{'imageId':1, 'url':'http://cont...'}, {'imageId':2...}, ...]
    except ValueError as err:
        print('Error occurs while loading the json file:  ' + err)
        return None
        
    urls = {int(img['imageId']): img['url'] for img in urls}
    return urls


def labelfromjson(file):
    """Return a matrix forming from labels of a json file in iMfashion competition.
       with 1st column of data is image id.
       the i-th column is the record of whether the image has label number i. (1 = yes. / 0 = no.)"""
    try:
        labels = json.load(file)
        labels = labels['annotations'] 
        # labels is a list: 
        # [{'imageId':1, 'labelId':['62','17','66'...]}, {'imageId':2...}, ...]
    except ValueError as err:
        print('Following error occurs while loading the json file:  ' + str(err))
        return None    
    
    # parsing labels, create a generator:
    f = lambda img: [int(img['imageId'])] + [int(i) for i in img['labelId']]
    imgs = map(f, labels)
    
    # building the matrix:
    n_col = 500  # default 500 columns
    need_cut = True
    matrix = numpy.zeros((len(labels), n_col), dtype=numpy.uint32)   
    
    # Create a numpy array according to labels --
    # 
    # Each row is the labels of an image --
    # the 1st column of data is the image id.
    # the i-th column is the record of whether the image has label number i. (1 = yes. / 0 = no.)
    # Ex. a row of [87, 1, 0, 0, 0, 1, 0, 0, 0, 1] means the image with id 87 is labeled with 1, 5 and 9.
    for img in imgs:
        try:
            matrix[img[0]-1, 0] = img[0]
            matrix[img[0]-1, img[1:]] = 1
        except IndexError:  # maybe 500 columns are not enough, then the following dirty code applies
            print('{0} columns are not enough, set # of columns to {1}'.format(n_col, max(img[1:])))
            n_col = max(img[1:]) + 1
            shape = lambda n_col, matrix: (1 if matrix.ndim==1 else matrix.shape[0], n_col-matrix.shape[1])
            matrix = numpy.concatenate((matrix, numpy.zeros(shape(n_col, matrix), dtype=numpy.uint32)), axis=1)
            matrix[img[0]-1,img[1:]] = 1
            need_cut = False
            
    # !magic steps for cutting the redudant zero blocks in matrix
    # see StackOverflow: 39465812/how-to-crop-zero-edges-of-a-numpy-array
    if need_cut:
        l = numpy.argwhere(matrix).max(axis=0)
        matrix = matrix[:l[0]+1, :l[1]+1]
        
    matrix = matrix[matrix[:, 0].argsort()]
    return matrix
      
def main():
    try:
        fr = open('data_train.json', 'r')
        fw = open('labels_train.pickle','wb')
        pickle.dump(labelfromjson(fr), fw)
    except OSError as err:
        print(err)
    finally:
        fr.close()
        fw.close()
    
if __name__=='__main__':
    main()
