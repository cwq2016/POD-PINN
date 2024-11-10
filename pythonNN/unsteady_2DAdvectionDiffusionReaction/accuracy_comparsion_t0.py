#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:06 2020

@author: wenqianchen
"""

import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')
sys.path.insert(0,'./NumSols')
from AdvectionDiffusionReaction import CustomedEqs, CustomedNet, DEVICE
import numpy as np
import torch
import os
from Cases_test_t0 import gen_testcases, NumSolsdir, M_Vec, FND, Ncut, useClosure, resultsdir
from Solver import Geometry
import matplotlib.pyplot as plt
from plotting import newfig,savefig
    
#resultsdir = 'results'
# validation space
file_Validation = NumSolsdir + '/' + 'PODG_Snapshots_Validation.npz'
datas  = np.load(file_Validation)
Snapshots_V = datas['Snapshots']
geo=Geometry()
alpha       = datas['parameters']
alpha_p     = alpha[:,0, 0:1]
Np,Nt       = alpha.shape[0:2]


PODG_prefix ="PODG_"
PODG_file_Validation = NumSolsdir + '/' + PODG_prefix+'Snapshots_Validation.npz'
datas  = np.load(PODG_file_Validation)
PODG_Snapshots_V = datas['Snapshots']
PODG_alpha       = datas['parameters']
PODG_alpha_p     = alpha[:,0, 0:1]

def CalculateError(testcases):
    name = testcases.name
    Vals = testcases.Vals
    Error=np.ones((len(Vals), len(M_Vec), 7))+1
    for case in testcases:
        M = case['M']; indM = M_Vec.index(M)
        ind = Vals.index(case[name])
        netfile = FND.NetFile(case)
        PODG_Projection_file= FND.PODGFile(case)
        filePOD = FND.PODFile(case)
        filePOD = filePOD.replace('Snapshots_POD_SVD', 'PODG_Snapshots_POD_SVD')
        roeqs = CustomedEqs(filePOD,case['M'])
        filePOD_PODG = filePOD
        roeqs_PODG = CustomedEqs(filePOD_PODG,case['M'])   
        
            
        # Projection && POD-G 
        lamda_G_PODG_useclosure=None
        lamda_G_PODG= None
        if os.path.isfile(PODG_Projection_file) and 0>1:
            datas = np.load(PODG_Projection_file)
            lamda_G           = datas['lamda_G']
            lamda_G_useclosure= datas['lamda_G_useclosure']
            lamda_G_PODG           = datas['lamda_G_PODG']
            lamda_G_PODG_useclosure= datas['lamda_G_PODG_useclosure']            
            lamda_Projection       = datas['lamda_Projection']
        else:
            # Projection
            lamda_Projection = roeqs.GetProjectionCoeff(Snapshots_V, roeqs.tgrid)
            ## POD-G from test interval 
            # POD-G without closure
            lamda_G            = roeqs.POD_G(alpha_p, lamda_Projection[:,0,:],useClosure=False)
            # POD-G use closure
            lamda_G_useclosure = roeqs.POD_G(alpha_p, lamda_Projection[:,0,:],useClosure=True )    
            
            ## POD-G from 0
            lamda_Projection_PODG = roeqs_PODG.GetProjectionCoeff(PODG_Snapshots_V, roeqs_PODG.tgrid)
            # POD-G without closure
            lamda_G_PODG            = roeqs_PODG.POD_G(PODG_alpha_p, lamda_Projection_PODG[:,0,:], useClosure=False)
            # POD-G use closure
            lamda_G_PODG_useclosure = roeqs_PODG.POD_G(PODG_alpha_p, lamda_Projection_PODG[:,0,:], useClosure=True)
            
            # save
            np.savez_compressed(PODG_Projection_file,**{'lamda_G_PODG':lamda_G_PODG,\
                                                        'lamda_G_PODG_useclosure':lamda_G_PODG_useclosure,\
                                                        'lamda_G':lamda_G,\
                                                        'lamda_G_useclosure':lamda_G_useclosure,\
                                                        'lamda_Projection':lamda_Projection,\
                                                        })

        # PODG error
        Ntcut = 20
        Error[ind,indM,0] = roeqs.GetError(lamda_Projection,   Snapshots_V, Ntcut)
        Error[ind,indM,3] = roeqs.GetError(lamda_G,            Snapshots_V, Ntcut)
        Error[ind,indM,4] = roeqs.GetError(lamda_G_useclosure, Snapshots_V, Ntcut)        
        #Ntcut = PODG_alpha.shape[1] - alpha.shape[1]
        #Ntcut = 150
        Error[ind,indM,5] = roeqs_PODG.GetError(lamda_G_PODG,            PODG_Snapshots_V, Ntcut)
        Error[ind,indM,6] = roeqs_PODG.GetError(lamda_G_PODG_useclosure, PODG_Snapshots_V, Ntcut)
        
        if M>30:
            continue
        Net =CustomedNet(Ncut, layers=None, oldnetfile=netfile, roeqs=roeqs).to(DEVICE)
        Net.loadnet(netfile)

        lamda_Net = Net(torch.tensor(alpha.reshape((-1,2))).float().to(DEVICE)).reshape((Np,Nt,-1))

        # POD_NN
        Nettype = case['Nettype']
        if Nettype == 'Label':
            Error[ind,indM,1] = roeqs.GetError(lamda_Net, Snapshots_V, Ntcut)
        # POD-Hybrid
        elif Nettype == 'Hybrid':
            #continue
            Error[ind,indM,2] = roeqs.GetError(lamda_Net, Snapshots_V, Ntcut)             
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
    newfig(width=1.3)
    for i in range(len(Vals)):
        namestr ="" # ', ' + '$N_s$=' + str(Vals[i])
        if name != 'SampleNum' and i ==0:
            plt.semilogy(M_Vec,Error[i,:,1], 'k-.',label='Projection'  )
            plt.semilogy(M_Vec,Error[i,:,0], 'r-.',label='POD-G      '  )
        elif name == 'SampleNum':
            plt.semilogy(M_Vec,Error[i,:,0], 'k-.'+symbols[i] ,label='Projection'+namestr  )
            plt.semilogy(M_Vec,Error[i,:,1], 'r-.'+symbols[i] ,label='PDNN'   +namestr  )
        #if not( name == 'NResi' ):
        #    plt.semilogy(M_Vec,Error[i,:,2], 'y:'+symbols[i]  ,label='PDNN'  +namestr  )
        plt.semilogy(M_Vec,Error[i,:,2], 'g-'+symbols[i]      ,label='PRNN'+namestr  )
        plt.semilogy(M_Vec,Error[i,:,3], 'b^'      ,label='PODG'         )
        plt.semilogy(M_Vec,Error[i,:,4], 'bo'      ,label='PODG + closure'  )
        #plt.semilogy(M_Vec,Error[i,:,5], 'y^'      ,label='PODG0'            )
        #plt.semilogy(M_Vec,Error[i,:,6], 'yo'      ,label='PODG0 + closure'  )

    plt.xlabel('$m$')
    plt.ylabel('Error')
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

    
    
