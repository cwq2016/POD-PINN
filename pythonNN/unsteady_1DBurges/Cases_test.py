#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:55:01 2020

@author: wenqianchen
"""
import sys,os
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')

from NN import train, DEVICE, train_options_default
import numpy as np
from Burges import CustomedEqs, CustomedNet

#
NetDepth=2
EPOCH   = int(2E4)
train_options_default['lamda'] = lambda epoch: 0.95**(epoch//200)
train_options_default['AdamScale']  = 0.9
train_options_default['early_stop'] = False
train_options_default['epoch_save'] = 1000
train_options_default["trainratio_label"] = 1

if len(sys.argv) == 12:
    ist1 = int(sys.argv[1])
    ien1 = int(sys.argv[2])
    ist2 = int(sys.argv[3])
    ien2 = int(sys.argv[4])
    ist3 = int(sys.argv[5])
    ien3 = int(sys.argv[6])
    ist4 = int(sys.argv[7])
    ien4 = int(sys.argv[8])
    ist5 = int(sys.argv[9])
    ien5 = int(sys.argv[10])
    resultsdir = sys.argv[11]
elif len(sys.argv) == 1:
    ist1 = 0
    ien1 = 1
    ist2 = 2
    ien2 = 3
    ist3 = 0
    ien3 = 1
    ist4 = 0
    ien4 = 1
    ist5 = 0
    ien5 = 1
    resultsdir = "results"
# test
else:
    raise Exception(' Wrong number of inputs')

if not os.path.isdir(resultsdir):
    os.mkdir(resultsdir)


NetSize_Vec       = [100]
NResi_Vec         = [2000]
SampleNum_Vec     = [5,10,20, 60, 100][ist1:ien1]
M_Vec             = list(range(3,20,3))[ist2:ien2]
RepeatInd_Vec     = list(range(10))[ist4:ien4]
WeightLog_Vec     = [-2, -1, 0, 1, 2, 3, 4, 5][ist5:ien5] # the default is zero, put in the first place.
Nettype_Vec       = ['Label','Hybrid'][ist3:ien3]
VarsRange_dict    = {\
                     'NetSize'  : (NetSize_Vec,   NetSize_Vec[0],  ), \
                     'NResi'    : (NResi_Vec,     NResi_Vec[0], ),\
                     'SampleNum': (SampleNum_Vec, SampleNum_Vec[0], ),\
                     }

Vars_dict         = {key:value[1] for key, value in VarsRange_dict.items()}
Vars_dict['M']=0; Vars_dict['Nettype']=0; Vars_dict['RepeatInd']=999; Vars_dict['WeightLog']=0
keys = list( Vars_dict.keys() )
    
def Dict2Str(dicti):
    describes = ''
    for name in keys:
        val = dicti[name]
        describes += '_'+name+''+str(val)
    return describes

class gen_testcases(object):
    def __init__(self, ControlVarName = 'SampleNum', RepeatFlag=True):
        self.fileprefix  = resultsdir +os.sep+'Burges'
        self.ControlVarName = ControlVarName
        self.RepeatFlag     = RepeatFlag
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
                    NRepeat=len(RepeatInd_Vec)
                    NWeightLog=len(WeightLog_Vec)
                    if not self.RepeatFlag:
                        NRepeat=1
                        NWeightLog=1
                    if Nettype != 'Hybrid':
                        NWeightLog=1
                    for RepeatInd in RepeatInd_Vec[:NRepeat]:
                        localdict['RepeatInd']=RepeatInd
                        if not self.RepeatFlag:
                            localdict['RepeatInd']=999
                        for WeightLog in WeightLog_Vec[:NWeightLog]:
                            localdict['WeightLog']=WeightLog
                            if Nettype != 'Hybrid' or not  self.RepeatFlag:
                                localdict['WeightLog']=0
                            yield localdict, Dict2Str(localdict)
    

    def Calculate(self):
        losshistory = {}
        for case in self:
            print(case[1])
            losshistory[case[1]+'train'], losshistory[case[1]+'test']=self.CaseSim(case)
        from scipy.io import savemat
        savemat('Test'+self.name+'_losshistory.mat', losshistory)
        return losshistory
        
    
    def CaseSim(self, case):
        netfile = self.fileprefix+'_'+            case[1]             +'.net'
        layers = [2, *[ case[0]['NetSize'] ]*NetDepth, case[0]['M']]
        roeqs = CustomedEqs(NSample=case[0]['SampleNum'], tend=2, xlen=1, Nt=100, Re_min = 200, Re_max=800)
        roeqs.setM(case[0]['M'])
        Net =CustomedNet(roeqs=roeqs, layers=layers, WeightLog=case[0]['WeightLog']).to(DEVICE)
        
        options = train_options_default.copy()
        options['EPOCH'] = EPOCH
        data_label = None; data_resi=None
        if case[0]['Nettype']=='Label':
            data_label = GetLabelData(roeqs)
            options['weiight_decay']=1E-10
        elif case[0]['Nettype']=='Hybrid':
            data_label,data_resi = GetHybridData(roeqs,case[0]['NResi'])
            options['weight_decay']=0
        options['NBATCH_label'] = 10
        options['NBATCH_resi' ] = 10
        # train the net and save loss history
        trainhistory, testhistory=train(Net, netfile, data_label=data_label, data_resi=data_resi, options=options)
        return trainhistory, testhistory

# Prepare trainning data
from pyDOE import lhs
from Normalization import Normalization
# projection  data
def GetLabelData(roeqs, Ninit=100):
    label_in  = roeqs.parameters
    label_out = roeqs.projections.T
    Nin = roeqs.design_space.shape[1]
    label_init_in_alpha = Normalization.Anti_Mapminmax(lhs(Nin-1,Ninit)*2-1,  roeqs.design_space[:,:-1])
    label_init_in_t     = np.zeros((label_init_in_alpha.shape[0],1))
    label_init_in       = np.concatenate((label_init_in_alpha, label_init_in_t), axis=1)
    label_init_out= np.zeros((Ninit, label_out.shape[1]))
    label_in  = np.concatenate((label_in,  label_init_in ), axis=0)
    label_out = np.concatenate((label_out, label_init_out), axis=0)
    return (label_in, label_out)
# Residual points
def GetResiData(roeqs, Np):
    Nin = roeqs.design_space.shape[1]
    Resi_in  = Normalization.Anti_Mapminmax(lhs(Nin,Np)*2-1,  roeqs.design_space)
    Re   = Resi_in[:,0:1]
    Binit    = roeqs.getBinit(Re).squeeze()
    source   = roeqs.getsource(Re)
    return (Resi_in, Binit, source)

# Residual points
def GetHybridData(roeqs, Np):
    return GetLabelData(roeqs), GetResiData(roeqs,Np) 

if __name__ == '__main__':
    gen_testcases('SampleNum').Calculate()
#    gen_testcases('NetSize'  ).Calculate()
#    gen_testcases('NResi'    ).Calculate()
    # test
    #casei = {'SampleNum':20, 'NetSize':20,'M':4,'Nettype':'Resi'}
    #Gcase = gen_testcases('SampleNum'  )
    #Gcase.CaseSim((casei, Gcase.Dict2Str(casei),))
    
    
    
