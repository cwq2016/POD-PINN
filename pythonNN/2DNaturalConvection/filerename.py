#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:20:16 2020

@author: wenqianchen
"""

import os

path = './NumSols/1E+04_1E+05and0.60_0.80and45_90/NaturalConvectionValidation'
for file in os.listdir(path):    #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
    if file[-2: ] == 'py':
        continue    #过滤掉改名的.py文件
    newname = file.replace('7_', '_')   #去掉空格
    os.rename(path + os.sep + file, path + os.sep + newname)