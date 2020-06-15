#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:55:01 2020

@author: wenqianchen
"""
import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')

from LidDriven import CustomedEqs, CustomedNet 
from Normalization import Normalization
from NN import train, DEVICE, train_options_default
import numpy as np

#
EPOCH   = int(3E4)
train_options_default['lamda'] = lambda epoch: 0.96**(epoch//200)
NumSolsdir = 'NumSols/100_500and60_120'
resultsdir = 'results'

# Control variables
#ist1 = int(sys.argv[1])
#ien1 = int(sys.argv[2])
#ist2 = int(sys.argv[3])
#ien2 = int(sys.argv[4])

ist1 = 0
ien1 = 6
ist2 = 0
ien2 = 3

M_Vec             = list(range(5,32,5))[ist1:ien1]
Nettype_Vec       = ['Label', 'Resi','Hybrid'][ist2:ien2]
SampleNum_Vec     = [30,60,100]
NetSize_Vec       = [30]
NResi_Vec         = [20000]
VarsRange_dict    = {'SampleNum': (SampleNum_Vec, SampleNum_Vec[0], ), \
                     'NetSize'  : (NetSize_Vec,   NetSize_Vec[0], ), \
                     'NResi': (NResi_Vec,   NResi_Vec[0], )
                     }

Vars_dict         = {key:value[1] for key, value in VarsRange_dict.items()}
Vars_dict['M']=0; Vars_dict['Nettype']=0;
keys = list( Vars_dict.keys() )
    

def Dict2Str(dicti):
    describes = ''
    for name in keys:
        val = dicti[name]
        if val == 80:
            val =10;
        describes += '_'+name+''+str(val)
    describes = 'LidDriven' + describes
    return describes

class gen_testcases(object):
    def __init__(self, ControlVarName = 'SampleNum'):
        self.ControlVarName = ControlVarName
        if  ControlVarName not in VarsRange_dict.keys():
            raise Exception('invalid var name')
        self.name = ControlVarName
        self.Vals = VarsRange_dict[ControlVarName][0]
    def __iter__(self):
        return self.gen_fun()

    def gen_fun(self):
        localdict = Vars_dict.copy()
        for val in VarsRange_dict[self.ControlVarName][0]:
            localdict[self.ControlVarName] = val
            for M in M_Vec:
                localdict['M'] = M
                for Nettype in Nettype_Vec:
                    localdict['Nettype']= Nettype
                    yield localdict, self.Dict2Str(localdict)

    def Calculate(self):
        losshistory = {}
        for case in self:
            losshistory[case[1]+'train'], losshistory[case[1]+'test']=self.CaseSim(case)
        from scipy.io import savemat
        savemat('Test'+self.name+'_losshistory.mat', losshistory)
        return losshistory
    
    def CaseSim(self, case):
        matfilePOD        = NumSolsdir  + '/'+'LidDrivenPOD.mat'
        matfileValidation = NumSolsdir  + '/'+'LidDrivenValidation.mat'
        netfile = resultsdir  + '/'+ case[1]+'.net'
        import os
        if os.path.isfile(netfile):
            pass
            #continue
        roeqs = CustomedEqs(matfilePOD,case[0]['SampleNum'],matfileValidation,case[0]['M'])
        layers = [2, *[ case[0]['NetSize'] ]*5, case[0]['M']]
        Net =CustomedNet(layers=layers, roeqs=roeqs).to(DEVICE)
        options = train_options_default.copy()
        options['EPOCH'] = EPOCH
        if case[0]['Nettype'] == 'Label':
            data = GetLabelData(roeqs)
            options['weight_decay']=1E-4
            options['NBATCH'] = 1
        elif case[0]['Nettype']== 'Resi':
            data = GetResiData(roeqs,case[0]['NResi'])
            options['weight_decay']=0
            options['NBATCH'] = case[0]['NResi']//1000
            #oldnetfile = netfile.replace('NettypeResi', 'NettypeLabel')
            #Net.loadnet(oldnetfile)
        elif case[0]['Nettype']== 'Hybrid':
            data = GetHybridData(roeqs,case[0]['NResi'])
            options['weight_decay']=0
            options['NBATCH'] = case[0]['NResi']//1000
        # train the net and save loss history
        trainhistory, testhistory=train(Net,data, netfile, options=options)
        return trainhistory, testhistory

# Prepare trainning data
# projection  data
def GetLabelData(roeqs):
    labeled_inputs = roeqs.parameters
    labeled_outputs= roeqs.projections.T 
    return (labeled_inputs, labeled_outputs, 'Label',0.9,)
# Residual points
def GetResiData(roeqs,Np):
    Nin = roeqs.design_space.shape[1]
    Resi_inputs  = Normalization.Anti_Mapminmax(np.random.rand(Np, Nin)*2-1,  roeqs.design_space)
    dummy = np.zeros((Np,roeqs.M))
    return (Resi_inputs, dummy, 'Resi',0.9,)
# Residual points
def GetHybridData(roeqs,Np):
    LabelData = GetLabelData(roeqs)
    ResiData  = GetResiData(roeqs,Np) 
    return (LabelData[0], LabelData[1],ResiData[0],ResiData[1],'Hybrid',ResiData[3],)

if __name__ == '__main__':
    gen_testcases('SampleNum').Calculate()
    
    
    
