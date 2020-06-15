#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:20:16 2020

@author: wenqianchen
"""

import os

for file in os.listdir('.'):    #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
    if file[-2: ] == 'py':
        continue   #过滤掉改名的.py文件
    name = file.replace(' ', '')   #去掉空格
    new_name = name[20: 30] + name[-4:]   #选择名字中需要保留的部分
    os.rename(file, new_name)