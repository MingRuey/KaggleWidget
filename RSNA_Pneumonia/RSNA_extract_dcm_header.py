# -*- coding: utf-8 -*-
"""
Created on 10/16/18
@author: MRChou

Scenario: Extract infomation from dcm file, especially the PA/AP view of x-ray.
"""

import pathlib
import pydicom as dcm


if __name__ == '__main__':
    img_path = '/rawdata/RSNA_Pneumonia/imgs_test/'
    path_gener = pathlib.Path(img_path).iterdir()

    for file in path_gener:
        print(str(file))

        ds = dcm.dcmread(str(file))
        print(ds.ViewPosition)
