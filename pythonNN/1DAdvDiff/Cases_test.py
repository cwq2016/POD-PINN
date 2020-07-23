#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:55:01 2020

@author: wenqianchen
"""
import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')
from AdvDiff1D import Net1DAdvDiff, Eqs1D 
from Normalization import Normalization
from NN import train, trainHybrid,DEVICE
import numpy as np
import torch
import torch.nn as nn

TRAINNING = True

matfile = 'problem1D.mat'
Mchooses = list(range(1,6,1))
Mchooses = Mchooses[0:]
Nettypes = ['Label', 'Resi']
Nettypes = Nettypes[0:2]
NResi   = int(5E3)

NPOD_G  = 100
trainratio = 0.9
train_options         ={'EPOCH':20000,\
                        'LR':0.01, \
                        'lamda': lambda epoch: 0.97**(epoch//200),\
                        'epoch_print': 1,\
                        'epoch_save':1000,\
                        'NBATCH':10,\
                        'weight_decay':0,\
                        }
losshistory = {}
for Mchoose in Mchooses:
    roeqs = Eqs1D(matfile, Mchoose)
    layers = [2, *[20]*3, Mchoose]
    Nin = roeqs.design_space.shape[1]
    Nout = Mchoose
    Net =Net1DAdvDiff(layers=layers, roeqs=roeqs).to(DEVICE)
    
    for Nettype in Nettypes:
        netfile = 'results/'+Nettype+'%d'%(Mchoose)+'.net'  
        if Nettype == 'Label':
            # projection  data
            labeled_inputs = roeqs.parameters
            labeled_outputs= (roeqs.projections.T - roeqs.proj_mean)/roeqs.proj_std
            options = train_options.copy()
            options['weight_decay']=1E-4
            options['NBATCH'] = 1
            trainhistory,testhistory =train(Net,(labeled_inputs, labeled_outputs, 'Label',trainratio,), netfile, options=options)
        elif Nettype == 'Resi':
            # Residual points
            Resi_inputs  = Normalization.Anti_Mapminmax(np.random.rand(NResi, Nin)*2-1,  roeqs.design_space)
            alpha1 = Resi_inputs[:,0:1]; alpha2=Resi_inputs[:,1:2]
            Resi_source = roeqs.getsourceNormal(alpha1, alpha2)
            options = train_options.copy()
            options['weight_decay']=0
            options['NBATCH'] = 10
            trainhistory,testhistory =train(Net,(Resi_inputs, Resi_source, 'Resi',trainratio,), netfile, options=options)
        elif Nettype == 'Resi + label':
            ## POD-G points
            alpha_G   = Normalization.Anti_Mapminmax(np.random.rand(NPOD_G,   Nin)*2-1,  roeqs.design_space)
            lamda_G = roeqs.POD_G(Mchoose, alpha_G)
            # Residual points
            Resi_inputs  = Normalization.Anti_Mapminmax(np.random.rand(NResi, Nin)*2-1,  roeqs.design_space)
            Resi_outputs = np.zeros((NResi, Nout))
            options = train_options.copy()
            options['weight_decay']=0
            options['NBATCH'] = 20
            trainhistory,testhistory =trainHybrid(Net,(alpha_G, lamda_G, Resi_inputs, Resi_outputs), netfile, options=options)
            
        losshistory[Nettype+'%d'%(Mchoose)+'train']=trainhistory
        losshistory[Nettype+'%d'%(Mchoose)+'test']=testhistory
        
from scipy.io import savemat
savemat('losshistory.mat', losshistory)