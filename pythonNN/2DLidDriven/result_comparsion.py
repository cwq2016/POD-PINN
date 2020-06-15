#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:06 2020

@author: wenqianchen
"""

import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')
from LidDriven import CustomedEqs, CustomedNet, DEVICE
from Cases_test import Dict2Str, NumSolsdir, resultsdir
from plotting import newfig,savefig
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec
import numpy as np
import torch

fielddir = 'fields_proj'
ValidationFalg     = 'Validation'
ValidationInd      = [23, 61,119]
casedict = {'SampleNum':100,\
            'NetSize':30,\
            'NResi':20000,\
            'M':25,\
            'Nettype':'Hybrid'}
case     =[casedict, Dict2Str(casedict)]
M = case[0]['M']
matfilePOD        = NumSolsdir  + '/'+'LidDrivenPOD.mat'
matfileValidation = NumSolsdir  + '/'+'LidDrivenValidation.mat'
netfile = resultsdir  + '/'+ case[1]+'.net'
roeqs = CustomedEqs(matfilePOD,case[0]['SampleNum'],matfileValidation,case[0]['M'])
Net =CustomedNet(layers=None, oldnetfile=netfile, roeqs=roeqs).to(DEVICE)
Net.loadnet(netfile)


alpha    = roeqs.ValidationParameters[ValidationInd ,:]
lamda    = Net(torch.tensor(alpha).float().to(DEVICE))
filename = fielddir + '/' +ValidationFalg
import os
from scipy.io import loadmat


if os.path.isfile(filename+'.mat'):
    Fields = loadmat(filename+'.mat')['Fields']
else:
    Fields   = roeqs.GetPredFields( alpha, lamda, filename)
    
Fields_Num = loadmat(matfileValidation)['Samples'][:,ValidationInd] 
umin, umax =  Fields_Num[1::6,:].min(), Fields_Num[1::6,:].max()
pmin, pmax =  Fields_Num[0::6,:].min(), Fields_Num[0::6,:].max()


def contourplot(ax, x,y, phi1, vmin, vmax, phi2, levels,title):
    sc1=ax.contourf(x,y,phi1, vmin=vmin, vmax=vmax,cmap='jet')
    sc2=ax.contour( x,y,phi2, levels=levels, colors=['k'])
    ax.set_title(title)
    ax.axis('off')
    return sc1,sc2

   
plt.close('all')
fig, _ =newfig(width=2, nplots=2/len(ValidationInd ))
gs = gridspec.GridSpec(2, len(ValidationInd))

for i in range(len(ValidationInd )):
    alphai = alpha[i:i+1,:]
    Fieldi = Fields[i]
    x, y, p, u, v, omega, psi = Fieldi[0], Fieldi[1], Fieldi[2], Fieldi[3], Fieldi[4], Fieldi[5], Fieldi[6]
    pNum = np.reshape( Fields_Num[0::6,i], x.shape)
    uNum = np.reshape( Fields_Num[1::6,i], x.shape)
    vNum = np.reshape( Fields_Num[2::6,i], x.shape)
    omegaNum = np.reshape( Fields_Num[4::6,i], x.shape)
    psiNum   = np.reshape( Fields_Num[5::6,i], x.shape)
    
    psi_max = psi.max(); psi_min = psi.min();
    psilevels = np.concatenate((np.linspace(psi_min, 0, 10),np.linspace(0,psi_max, 3)[1:], ))
    
    title = '$\\boldsymbol{\\mu}=(%3d,%3d)$'%(alphai[0,0], alphai[0,1])
    sc1,sc2 = contourplot(plt.subplot(gs[0, i]), x, y, p,   pmin, pmax,  psi   , psilevels,title+', POD-PINN2')
    sc1,sc2 = contourplot(plt.subplot(gs[1, i]), x, y, pNum,pmin, pmax,  psiNum, psilevels,title+', PS')
    
#colorbar 左 下 宽 高 
l = 0.93
b = 0.1
w = 0.015
h = 0.8
#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h] 
cbar_ax = fig.add_axes(rect) 
plt.colorbar(sc1, cax=cbar_ax)

savefig('fig/ResultComparsion')
