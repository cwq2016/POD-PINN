# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:31:01 2020

@author: 56425
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from NN import NN
from Normalization import Normalization

train_ratio    = 0.8
TRAINING     = True
EPOCH        = 3000  
HIDDENLAYERS = [20]*5
LR           = 0.01  # learning rate

plt.close('all')
# dataset_input  = loadmat('./Soboldata/InputX.mat')['InputX'];
# dataset_input  = np.delete(dataset_input,-1,1)
# dataset_output = loadmat('./Soboldata/OutputY.mat')['OutputY'];

dataset_input  = loadmat('./x.mat')['sample_x'];
dataset_output = loadmat('./y.mat')['sample_y'];

dataset_input, mapping_input  = Normalization.Normalize(dataset_input)
dataset_output, mapping_output = Normalization.Normalize(dataset_output)

Nout = dataset_output.shape[1]
Net = [None] * Nout
Netfile = ['Net%d.net'%i for i in range(Nout)]
loss_history = [None] * Nout;
for io in range(2,dataset_output.shape[1]):
    # fig= plt.figure( figsize=(6, 6) )
    # plt.axis('on')
    Net[io] = NN(dataset_input, dataset_output[:,io:io+1], train_ratio, HIDDENLAYERS = [20]*3, NBATCH=5)
    if TRAINING:
        loss_history[io] = zip( Net[io].train(Netfile[io], EPOCH= 3000, LR=0.01) )
    else:
        Net[io].loadnet(Netfile[io])









