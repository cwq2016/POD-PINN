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
import numpy as np
import torch
import os
from scipy.io import savemat, loadmat
from Cases_test import gen_testcases, NumSolsdir, SampleNum_Vec, M_Vec, resultsdir
import matplotlib.pyplot as plt
import matplotlib as rc
from plotting import newfig,savefig
    
# validation space
matfile = NumSolsdir  + '/'+'Burges1D_SampleNum='+str(SampleNum_Vec[0])+'.mat'
roeqs   = CustomedEqs(matfile, M_Vec[0])
alpha1  = np.linspace(roeqs.design_space[0,0],roeqs.design_space[1,0],101)
alpha2  = np.linspace(roeqs.design_space[0,1],roeqs.design_space[1,1],101)
alpha1, alpha2 = np.meshgrid(alpha1,alpha2);
alpha = np.stack((alpha1,alpha2), axis=2).reshape(-1, 2)


def CalculateError(testcases):
    name = testcases.name
    Vals = testcases.Vals
    Error=-np.ones((len(Vals), len(M_Vec), 5))
    for case in testcases:
        M = case[0]['M']; indM = M_Vec.index(M)
        ind = Vals.index(case[0][name])
        matfile = NumSolsdir  + '/'+'Burges1D_SampleNum='+str(case[0]['SampleNum'])+'.mat'
        netfile = resultsdir  + '/'+            case[1]             +'.net'
        PODGfile= resultsdir  + '/PODG/'+       case[1]             +'.mat'
        roeqs = CustomedEqs(matfile, case[0]['M'])
        print(case[1])
        # POD-G
        if os.path.isfile(PODGfile):
            lamda_G = loadmat(PODGfile)['lamda_G']
        else:
            lamda_G = roeqs.POD_G(M, alpha)
            savemat(PODGfile, {'lamda_G':lamda_G})
        Error[ind,indM,0] = roeqs.GetError(alpha,lamda_G)
        
        # Projection
        Error[ind,indM,4] =  roeqs.GetProjError(alpha)
        
#        continue
    
        Net =CustomedNet(layers=None, oldnetfile=netfile, roeqs=roeqs)
        Net.loadnet(netfile)

        lamda_Net = Net(torch.tensor(alpha).float().to(DEVICE))
        # POD_NN
        Nettype = case[0]['Nettype']
        if Nettype == 'Label':
            Error[ind,indM,1] = roeqs.GetError(alpha, lamda_Net) 
        # POD-PINN
        elif Nettype == 'Resi':
            Error[ind,indM,2] = roeqs.GetError(alpha, lamda_Net) 
            
        elif Nettype == 'Hybrid':
            Error[ind,indM,3] = roeqs.GetError(alpha, lamda_Net)             
    return Error
#字符|类型 | 字符|类型
#---|--- | --- | ---
#`  '-'	`| 实线 | `'--'`|	虚线
#`'-.'`|	虚点线 | `':'`|	点线
#`'.'`|	点 | `','`| 像素点
#`'o'`	|圆点 | `'v'`|	下三角点
#`'^'`|	上三角点 | `'<'`|	左三角点
#`'>'`|	右三角点 | `'1'`|	下三叉点
#`'2'`|	上三叉点 | `'3'`|	左三叉点
#`'4'`|	右三叉点 | `'s'`|	正方点
#`'p'`	| 五角点 | `'*'`|	星形点
#`'h'`|	六边形点1 | `'H'`|	六边形点2 
#`'+'`|	加号点 | `'x'`|	乘号点
#`'D'`|	实心菱形点 | `'d'`|	瘦菱形点 
#`'_'`|	横线点 | |

#字符 | 颜色
#-- | -- 
#`‘b’`|	蓝色，blue
#`‘g’`|	绿色，green
#`‘r’`|	红色，red
#`‘c’`|	青色，cyan
#`‘m’`|	品红，magenta
#`‘y’`|	黄色，yellow
#`‘k’`|	黑色，black
#`‘w’`|	白色，white
# visualization
def VisuError(Error, testcases, Savetofile=False):
    name = testcases.name
    Vals = testcases.Vals
    symbols = ('^','o', 's','D','p','H',)
    lines   = ('-', '--',':', '-.','.',)
    #plt.close('all')i
    if name == 'SampleNum':
        tmp = '$N_s$'
        width = 1.3
    elif name =='NetSize':
        tmp = '$n_H$'
        width = 1
    elif name == 'NResi':
        tmp = '$N_{Resi}$'
        width = 1
    newfig(width=width)
    for i in range(len(Vals)):
        namestr = ', ' + tmp+'=' + str(Vals[i])
        if name != 'SampleNum' and i ==0:
            plt.semilogy(M_Vec,Error[i,:,4], 'k-.',label='Projection'  )
            plt.semilogy(M_Vec,Error[i,:,0], 'r-.',label='POD-G      '  )
        elif name == 'SampleNum':
            plt.semilogy(M_Vec,Error[i,:,4], 'k-.'+symbols[i] ,label='Projection'+namestr  )
            plt.semilogy(M_Vec,Error[i,:,0], 'r-.'+symbols[i] ,label='POD-G'   +namestr  )
        if not( name == 'NResi' ):
            plt.semilogy(M_Vec,Error[i,:,1], 'y:'+symbols[i]  ,label='PDNN'  +namestr  )
        plt.semilogy(M_Vec,Error[i,:,2], 'g--'+symbols[i]     ,label='PINN'+namestr  )
        plt.semilogy(M_Vec,Error[i,:,3], 'g-'+symbols[i]      ,label='PRNN'+namestr  )

    plt.xlabel('$m$')
    plt.ylabel(r'Error $\varepsilon$')
    #plt.title(name)
    plt.legend(loc="lower left", ncol=1, handlelength=3)
    plt.show()
    
    if Savetofile:
        savefig("fig/ErrorComparsion_"+name)


if __name__ == "__main__":
    
    Errors = loadmat('Errors.mat')
    Error_SampleNum = Errors['Error_SampleNum']
    Error_NetSize = Errors['Error_NetSize']  
    Error_NResi   = Errors['Error_NResi']
    
    plt.close('all')
    Savetofile = True
    TestSampleNum = gen_testcases('SampleNum')
    #Error_SampleNum0 = CalculateError(TestSampleNum);
    VisuError(Error_SampleNum, TestSampleNum, Savetofile)
    
    TestNetSize = gen_testcases('NetSize')
    #Error_NetSize = CalculateError(TestNetSize);
    VisuError(Error_NetSize, TestNetSize, Savetofile)
    
    
    TestNResi  = gen_testcases('NResi')
    #Error_NResi = CalculateError(TestNResi);
    VisuError(Error_NResi, TestNResi, Savetofile)    

    
    
    

    
    