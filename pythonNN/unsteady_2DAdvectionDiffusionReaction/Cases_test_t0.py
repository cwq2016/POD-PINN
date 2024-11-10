#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:55:01 2020

@author: wenqianchen
"""
import sys, os
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')

from AdvectionDiffusionReaction import CustomedEqs, CustomedNet
from Normalization import Normalization
from NN import train, DEVICE, train_options_default
import numpy as np


#

EPOCH   = int(2E4)
train_options_default['lamda'] = lambda epoch: 0.95**(epoch//200)
train_options_default['AdamScale']  = 0.9
train_options_default['early_stop'] = False
train_options_default['epoch_save'] = 1000
train_options_default["trainratio_label"] = 1
NumSolsdir = './NumSols/0.05_0.50'

Ncut =0; # cut the first Ncut time steps to guarantee smooth
coarse_time_resi = 1; # coarse time grid for generating resi points
useClosure = True

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

M_Vec             = list(range(5,31,5))[ist1:ien1]
Nettype_Vec       = ['Label', 'Hybrid'][ist2:ien2]
RepeatInd_Vec     = list(range(10))[ist4:ien4]

SampleNum_Vec     = [10,20][ist3:ien3]
NetSize_Vec       = [100]
NResi_Vec         = [2048]
WeightLog_Vec     = [-2,-1,0,1,2][ist5:ien5] 
VarsRange_dict    = {'SampleNum': (SampleNum_Vec, SampleNum_Vec[0], ), \
                     'NetSize'  : (NetSize_Vec,   NetSize_Vec[0], ), \
                     'NResi': (NResi_Vec,   NResi_Vec[0], ),\
                     'WeightLog':(WeightLog_Vec, WeightLog_Vec[0],),\
                     'RepeatInd':(RepeatInd_Vec, RepeatInd_Vec[0],),\
                     }

Vars_dict         = {key:value[1] for key, value in VarsRange_dict.items()}
Vars_dict['M']=0; Vars_dict['Nettype']=0; Vars_dict['RepeatInd']=999; Vars_dict['WeightLog']=0
keys = list( Vars_dict.keys() )


class FileNameDefine():
    def __init__(self,RootName):
        self.RootName = RootName
    def NetFile(self,case):
        NetFile = ''
        for name in keys:
            val = case[name]
            NetFile += '_'+name+''+str(val)
        NetFile = self.RootName + NetFile
        return resultsdir  + os.sep + NetFile + ".net"
    def PODFile(self,case):
        return NumSolsdir  + os.sep+'Snapshots_POD_SVD%d.npz' % case['SampleNum'] 
    def LossHistory(self,case):
        return self.NetFile(case) +  '_LossHistory.npz'
    def NetResidual(self,case):
        path = resultsdir + os.sep + 'ResidualNet' 
        if not os.path.exists(path): 
            os.mkdir(path)
        return  path + os.sep + self.RootName + '_ResidualNet' + '_SampleNum%d_M%d.net'%(case['SampleNum'],case['M'])
    def PODGFile(self,case):
        path = resultsdir + os.sep + 'PODG'  
        if not os.path.exists(path): 
            os.mkdir(path)
        return  path + os.sep + self.RootName + '_PODG_Projection' + '_SampleNum%d_M%d.npz'%(case['SampleNum'],case['M'])
    
    def AllName(self,case):
        return self.NetFile(case),  self.PODFile(case),  self.LossHistory(case)
    
FND  = FileNameDefine('NaturalConvection')
    
class gen_testcases(object):
    def __init__(self, ControlVarName = 'SampleNum', RepeatFlag=True):
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
                            yield localdict

    def Calculate(self):
        for case in self:
            losshistory = {}
            losshistory['train'], losshistory['test']=self.CaseSim(case)
            #np.savez_compressed('Test'+self.name+'_losshistory.mat', losshistory)
        
    def CaseSim(self, case):
        netfile, filePOD  = FND.AllName(case)[0:2]
        filePOD = filePOD.replace('Snapshots_POD_SVD', 'PODG_Snapshots_POD_SVD')
        roeqs  = CustomedEqs(filePOD, case['M'])
        layers = [2, *[ case['NetSize'] ]*2, case['M']]
        Net =CustomedNet(Ncut, layers=layers, roeqs=roeqs, WeightLog=case['WeightLog']).to(DEVICE)
        options = train_options_default.copy()
        options['EPOCH'] = EPOCH
        data_label = None; data_resi=None
        
        if case['Nettype'] == 'Label':
            data_label = GetLabelData(roeqs, Ninit=100)
            options['weight_decay']=1E-10
        elif case['Nettype']== 'Hybrid':
            data_label,data_resi = GetHybridData(roeqs,case['NResi'], Ninit=100 )
            options['weight_decay']=0
        options['NBATCH_label'] = 10
        options['NBATCH_resi' ] = 10
        # train the net and save loss history
        trainhistory, testhistory=train(Net, netfile, data_label=data_label, data_resi=data_resi, options=options)
        return trainhistory, testhistory

# Prepare trainning data
from pyDOE import lhs  
# projection  data
def GetLabelData(roeqs, Ninit=100):
    label_in = roeqs.parameters.reshape((-1,roeqs.Nt,2)).reshape((-1,2))
    label_out= roeqs.projections.reshape((-1,roeqs.Nt,roeqs.M)).reshape((-1,roeqs.M)) 
    # if global prediction
    if abs(roeqs.design_space[0,-1])<1E-10:
        Nin = roeqs.design_space.shape[1]
        if Ninit>0:
            label_init_in_alpha = Normalization.Anti_Mapminmax(lhs(Nin-1,Ninit)*2-1,  roeqs.design_space[:,:-1])
            label_init_in_t     = np.zeros((label_init_in_alpha.shape[0],1))
            label_init_in       = np.concatenate((label_init_in_alpha, label_init_in_t), axis=1)
            label_init_out= np.zeros((Ninit, label_out.shape[1]))
            label_in  = np.concatenate((label_in,  label_init_in ), axis=0)
            label_out = np.concatenate((label_out, label_init_out), axis=0)        
    #return (labeled_inputs.reshape((-1,3)), labeled_outputs.reshape((-1,roeqs.M)),)
    # lamda_init = roeqs.POD_G(roeqs.parameters[:,0,:-1],useClosure=useClosure, Nt=Ncut+1)[:,-1:,:]
    # lamda_init = np.tile(lamda_init, (1,roeqs.Nt,1))
    #lamda_init = np.tile(labeled_outputs[:,Ncut:Ncut+1,:],  (1,roeqs.Nt,1))
    bundle     =  (label_in, label_out )
    return bundle
# Residual points
def GetResiData_rand(roeqs,Nresi, case=None):
    Nin = roeqs.design_space.shape[1]
    alpha  = Normalization.Anti_Mapminmax(lhs(Nin, Nresi)*2-1,  roeqs.design_space)
    return (alpha,)

def GetResiData_tensor(roeqs,Nresi,coarse=coarse_time_resi):
    t        = roeqs.tgrid[Ncut::coarse,0]
    Nt       = t.shape[0]
    Nin      = roeqs.design_space.shape[1]
    Np       = Nresi//Nt
    alpha_p  = Normalization.Anti_Mapminmax(lhs(Nin-1,Np)*2-1,  roeqs.design_space[:,:-1])    
    alpha_p  = np.tile(alpha_p[:,None,:], (1, Nt, 1))
    t        = np.tile(t[None,:,None], (Np,1,1))
    alpha    = np.concatenate((alpha_p, t), axis=2)
    bundle   = (alpha,)
    return (e[:,:,:] for e in bundle)


        

# Hybrid data
def GetHybridData(roeqs, Nresi, Ninit=100):
    return GetLabelData(roeqs, Ninit), GetResiData_rand(roeqs,Nresi)



if __name__ == '__main__':
    gen_testcases('SampleNum').Calculate()
    
    
    
