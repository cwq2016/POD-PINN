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
import numpy as np
import torch
import os
from scipy.io import savemat, loadmat
from Cases_test import gen_testcases, NumSolsdir, SampleNum_Vec, M_Vec, resultsdir
import matplotlib.pyplot as plt
import matplotlib as rc
from plotting import newfig,savefig
    
#resultsdir = 'results'
# validation space
matfilePOD        = NumSolsdir  + '/'+'LidDrivenPOD.mat'
matfileValidation = NumSolsdir  + '/'+'LidDrivenValidation.mat'
roeqs   = CustomedEqs(matfilePOD, 3,matfileValidation, M_Vec[0])
alpha   = roeqs.ValidationParameters


def CalculateError(testcases):
    name = testcases.name
    Vals = testcases.Vals
    Error=np.ones((len(Vals), len(M_Vec), 5))+1
    for case in testcases:
        M = case[0]['M']; indM = M_Vec.index(M)
        ind = Vals.index(case[0][name])
        netfile = resultsdir  + '/'+            case[1]             +'.net'
        PODGfile= resultsdir  + '/PODG/'+       case[1]             +'.mat'
        roeqs = CustomedEqs(matfilePOD, case[0]['SampleNum'],matfileValidation,case[0]['M'])
        print(PODGfile)
        # POD-G
        if os.path.isfile(PODGfile) and 0>1:
            lamda_G = loadmat(PODGfile)['lamda_G']
        else:
            lamda_G = roeqs.POD_Gfsolve(alpha, roeqs.lamda_proj)
            savemat(PODGfile, {'lamda_G':lamda_G})
        # PODG error
        Error[ind,indM,0] = roeqs.GetError(lamda_G)[1]
        # Projection error
        Error[ind,indM,4] = roeqs.ProjError[1]
        
        #continue
        Net =CustomedNet(layers=None, oldnetfile=netfile, roeqs=roeqs).to(DEVICE)
        Net.loadnet(netfile)

        lamda_Net = Net(torch.tensor(alpha).float().to(DEVICE))
        # POD_NN
        Nettype = case[0]['Nettype']
        if Nettype == 'Label':
            Error[ind,indM,1] = roeqs.GetError(lamda_Net)[1]
        # POD-PINN
        elif Nettype == 'Resi':
            Error[ind,indM,2] = roeqs.GetError(lamda_Net)[1] 
        # POD-PINNHybrid
        elif Nettype == 'Hybrid':
            Error[ind,indM,3] = roeqs.GetError(lamda_Net)[1]             
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
    newfig(width=1.4)
    for i in range(len(Vals)):
        namestr = ', ' + '$N_s$=' + str(Vals[i])
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
    plt.ylabel('Error')
    #plt.title(name)
    plt.legend(loc="lower left", ncol=1, handlelength=3)
    plt.show()

    if Savetofile:
        savefig("fig/ErrorComparsion_"+name)


if __name__ == "__main__":
    
#    Errors1 = loadmat('ErrorsHybrid_100_200and60_120.mat')['Error_SampleNum']
#    #Errors2 = loadmat('ErrorsOnlyResi_100_200and60_120.mat')['Error_SampleNum']
#    Errors  = loadmat('Errors_proj'                        )['Error_SampleNum']
#    Errors[:,:,3] = Errors1[:,2:,2]
#    
#    savemat('Errors_backupFor100_200and60_120.mat',{'Error_SampleNum':Errors})
    
    Error_SampleNum = loadmat('Errors_all.mat')['Error_SampleNum']
    TestSampleNum   = gen_testcases('SampleNum')
    Savetofile = True   
#    Error_SampleNum = CalculateError(TestSampleNum);
    VisuError(Error_SampleNum, TestSampleNum, Savetofile) 
    
    

    
    