#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:06 2020

@author: wenqianchen
"""

import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')
from Burges import CustomedEqs, CustomedNet, DEVICE
import numpy as np
import torch
from scipy.io import savemat, loadmat
from Cases_test import gen_testcases, SampleNum_Vec, M_Vec, resultsdir
import matplotlib.pyplot as plt
import matplotlib as rc
from plotting import newfig,savefig
    
# validation space
roeqs   = CustomedEqs()
Re      = np.linspace(roeqs.design_space[0,0], roeqs.design_space[1,0],11)[:,None]
alpha   = np.stack(np.meshgrid(Re,roeqs.tgrid), axis = 2).transpose((1,0,2)).reshape((-1,2))

def CalculateError(testcases):
    name = testcases.name
    Vals = testcases.Vals
    fileprefix = testcases.fileprefix
    Error=-np.ones((len(Vals), len(M_Vec), 4))
    for case in testcases:
        M = case[0]['M']; indM = M_Vec.index(M)
        ind = Vals.index(case[0][name])
        roeqs = CustomedEqs(NSample=case[0]['SampleNum'])
        roeqs.setM(case[0]['M'])
        netfile = fileprefix  + '_'+            case[1]             +'.net'
        print(case[1])
        # Projection
        Error[ind,indM,0] =  roeqs.GetProjError(Re)
        
        # POD-G
        Error[ind,indM,1] = roeqs.GetPODGError(Re)
        Nettype = case[0]['Nettype']
        if Nettype == 'Hybrid' and case[0]['SampleNum']>20:
            continue
        Net =CustomedNet(layers=None, oldnetfile=netfile, roeqs=roeqs).to(DEVICE)
        Net.loadnet(netfile)

        lamda_Net = Net(torch.tensor(alpha).float().to(DEVICE))\
                   .reshape((-1, roeqs.Nt+1, roeqs.M))
        # POD_NN
        # Nettype = case[0]['Nettype']
        if Nettype == 'Label':
            Error[ind,indM,2] = roeqs.GetglobalError(Re,lamda_Net)           
        elif Nettype == 'Hybrid':
            Error[ind,indM,3] = roeqs.GetglobalError(Re,lamda_Net)         
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
    elif name == 'NResi':
        tmp = '$N_{Resi}$'
        width = 1
    newfig(width=width)
    for i in range(len(Vals)):
        namestr = ', ' + tmp+'=' + str(Vals[i])
        if name != 'SampleNum' and i ==0:
            plt.semilogy(M_Vec,Error[i,:,0], 'k-.',label='Projection'  )
            plt.semilogy(M_Vec,Error[i,:,1], 'r-.',label='POD-G      '  )
        elif name == 'SampleNum':
            plt.semilogy(M_Vec,Error[i,:,0], 'k-.'+symbols[i] ,label='Projection'+namestr  )
            plt.semilogy(M_Vec,Error[i,:,1], 'r-.'+symbols[i] ,label='POD-G'   +namestr  )
        if not( name == 'NResi' ):
            plt.semilogy(M_Vec,Error[i,:,2], 'y:'+symbols[i]  ,label='PDNN'  +namestr  )
        plt.semilogy(M_Vec,Error[i,:,3], 'g-'+symbols[i]      ,label='PRNN'+namestr  )

    plt.xlabel('$m$')
    plt.ylabel(r'Error $\varepsilon$')
    #plt.title(name)
    plt.legend(loc="lower left", ncol=1, handlelength=3)
    plt.show()
    
    if Savetofile:
        savefig("fig/ErrorComparsion_"+name)


if __name__ == "__main__":
    loadError = False
    Savetofile = True 
    ErrorFile = 'Errors'+resultsdir+'.npz'
    TestSampleNum   = gen_testcases('SampleNum', False)
    if loadError: 
        Error_SampleNum = np.load(ErrorFile)['Error_SampleNum']
        VisuError(Error_SampleNum, TestSampleNum, Savetofile)
    else:
        Error_SampleNum = CalculateError(TestSampleNum);
        print("Error:")
        print(Error_SampleNum)
        #VisuError(Error_SampleNum, TestSampleNum, Savetofile)
        np.savez_compressed(ErrorFile, **{'Error_SampleNum':Error_SampleNum})        
    
    
    

    
    
