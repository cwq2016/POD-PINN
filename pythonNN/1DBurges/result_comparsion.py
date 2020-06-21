#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:06 2020

@author: wenqianchen
"""

import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')
from Net1Dburges import CustomedEqs, CustomedNet, DEVICE
from Cases_test import Dict2Str, NumSolsdir, resultsdir
from plotting import newfig,savefig
import matplotlib.pyplot as plt  
import numpy as np
import torch


matfile = 'problem1D.mat'
casedict = {'SampleNum':10,\
            'NetSize':20,\
            'NResi':5000,\
            'M':9,\
            'Nettype':'Hybrid'}
case     =[casedict, Dict2Str(casedict)]
M = case[0]['M']
matfile = NumSolsdir  + '/'+'Burges1D_SampleNum='+str(case[0]['SampleNum'])+'.mat'
netfile = resultsdir  + '/'+            case[1]             +'.net'
roeqs = CustomedEqs(matfile, case[0]['M'])
Net =CustomedNet(layers=None, oldnetfile=netfile, roeqs=roeqs).to(DEVICE)
Net.loadnet(netfile)

alpha    = np.array( [[  1,  9], \
                      [  5, 10], \
                      [  3,  7], \
                      [  10, 5]] )

symbols = ('^','o', 's','D','p','H',)
lines   = ('-', '--',':', '-.','.',)
colors  = ('r','g','b','g','y',)
plt.close('all')
newfig(width=1)
for k in range(2):
    for i in range(alpha.shape[0]):
        alphai    = alpha[i:i+1,:]
        lamdai    = Net(torch.tensor(alphai).float().to(DEVICE))
        phi_proj  = np.matmul(lamdai, roeqs.Modes.T)
        phi_Exact = roeqs.phix(roeqs.xgrid.T,alphai[:,0:1], alphai[:,1:2])
    
        #plot
        name = '$\\boldsymbol{\\mu}=(%0.1f,%0.1f)$'%(alphai[0,0],alphai[0,1])
        if k == 0:
            plt.plot(roeqs.xgrid, phi_Exact.T, colors[i]+lines[0]  ,label='PS'           ,markersize=6  )
        else:
            plt.plot(roeqs.xgrid, phi_proj.T , colors[i]+symbols[i],label='PRNN, '+name,markersize=6  )

plt.xlabel('$x$')
plt.ylabel('$\phi$')
#plt.title(resultsdir)
plt.legend(loc="upper left", ncol=2, handlelength=2, columnspacing=1)
plt.show()
savefig('fig/ResultComparsion')