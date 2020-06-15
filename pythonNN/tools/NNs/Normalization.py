# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:50:24 2020

@author: 56425
"""
import numpy as np
class Normalization():
    def Mapstatic(data):
        data_mean = np.mean(data, 0)
        data_std  = np.std(data, 0)
        data = (data - data_mean)/data_std
        mapping = (data_mean, data_std,)
        return data, mapping
    
    def Anti_Mapstatic(data,mapping):
        data_mean, data_std = mapping
        data = data*data_std + data_mean
        return data
    def Mapminmax(data,bounds):
        lb = bounds[0:1,:]
        ub = bounds[1:2,:]
        data = (data-lb)/(ub-lb)*2-1
        return data
    def Anti_Mapminmax(data,bounds):
        lb = bounds[0:1,:]
        ub = bounds[1:2,:]
        data = (data+1)/2*(ub-lb) +lb
        return data