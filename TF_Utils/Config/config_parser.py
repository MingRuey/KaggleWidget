# -*- coding: utf-8 -*-
"""
Created on 9/15/18
@author: MRChou

Scenario: Read .config file for tranning network in tensorflow
"""

import pathlib
import configparser

CONFIG = '/home/mrchou/code/KaggleWidget/TF_Utils/Config/examples/pipeline.cfg'


class _TrainCfg:
    pass

class _ModelCfg:
    pass


class Config:

    def __init__(self, cfg_path):
        pass


def parse_config(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    return cfg


if __name__ == '__main__':
    print(list(parse_config(CONFIG)['TRAIN_INPUT']))
