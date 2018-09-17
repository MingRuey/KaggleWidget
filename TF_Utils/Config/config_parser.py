# -*- coding: utf-8 -*-
"""
Created on 9/15/18
@author: MRChou

Scenario: Read .config file for tranning network in tensorflow
"""

import configparser


def parse_config(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    return cfg


if __name__ == '__main__':
    pass
