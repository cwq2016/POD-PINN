#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:20:16 2020

@author: wenqianchen
"""

import os

path = './results'
for file in os.listdir(path):    #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
    if file[-2: ] == 'py':
        continue    #过滤掉改名的.py文件
    newname = file.replace('=', '')   #去掉空格
    os.rename(path + os.sep + file, path + os.sep + newname)