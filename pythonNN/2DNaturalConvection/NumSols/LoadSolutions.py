# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:34:17 2020

@author: 56425
"""


from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import re
import os

def LoadSolutions(solroot, solname, samples_file,design_space,outputfilename):
    df = pd.read_csv(solroot+"/"+samples_file,skiprows=6,header=None, sep="\t");
    parameters = df.values[:,1:4]
    Nsample = parameters.shape[0]
    Samples = []
    vl, vr =[], [];
    k=0
    Ind_NoError=[]
    for i in range(Nsample):
        samplesolfile = solroot+"/"+solname+ "_%d/OUTPUT/Time=0.100/RESULT.plt"%(i+1);
        if os.path.isfile(samplesolfile):
            with open(samplesolfile,'r') as f:
                line = f.readline();
                line = f.readline();
                line = f.readline();
            Ind_NoError.append(i)                
        else:
            print(i+1)
            k=k+1
            continue
        FieldShape =np.array([ int(num) for num in re.findall(r"\d+",line)])
        df = pd.read_csv(samplesolfile,skiprows=3,header=None, sep="\s+", usecols=[0,1,2,3,4,5,6,7]);
        snapshot = np.reshape( df.values, (*FieldShape, 8, ), "F")
        # define the center as reference point for pressure
        snapshot[:,:,2]=snapshot[:,:,2] - snapshot[(FieldShape[0]-1)//2, (FieldShape[1]-1)//2*0, 2] 
        # address the singular problem when theta=90
        if snapshot[ 4,24,4] >0 : 
            snapshot[:,:,2:] =  snapshot[:,::-1,2:]
            snapshot[:,:,[4,6,7] ] *= -1
        Samples.append( snapshot[::-1,::-1,2:].reshape((-1), ))
        vl.append(snapshot[ 4,24,4])
        vr.append(snapshot[-5,24,4])
    Samples = np.stack(tuple(Samples), axis=1)
    print(k)
    savemat(solroot+"/"+outputfilename, {"FieldShape":FieldShape,\
                                          "Samples":Samples,\
                                          "parameters":parameters[Ind_NoError,:],\
                                          "design_space":design_space})
    return np.stack((vl, vr), axis=1)
    
if __name__ == "__main__":
    design_space = np.array([[1E5,0.6,0],[1E6,0.8,180]])
    design_space = np.array([[1E5,0.6,0],[3E5,0.8,90]])
    design_space = np.array([[1E5,0.65,60],[5E5,0.75,120]])
    design_space = np.array([[1E5,0.6,60],[5E5,0.8,90]])
#    design_space = np.array([[1E5,0.6, 0],[5E5,0.8,90]])
   # design_space = np.array([[1E5,0.65,60],[4E5,0.75,90]])
    #design_space = np.array([[1E5,0.70,60],[5E5,0.71,61]])
    #design_space = np.array([[1E5,0.70,60],[1+1E5,0.71,90]])
    design_space = np.array([[1E4,0.60,45],[1E5,0.80,90]])  
    #design_space = np.array([[1E4,0.60,60],[1E5,0.80,90]])  
    root='%0.0E_%0.0Eand%0.2f_%0.2fand%d_%d'%(design_space[0,0],design_space[1,0], \
                                  design_space[0,1],design_space[1,1], \
                                  design_space[0,2],design_space[1,2],)
#    ind = 7
#    root = 'others'
    
    solname ="NaturalConvectionPOD/NaturalConvectionPOD"
    samples_file = "NaturalConvectionPOD.txt"
    outputfilename = "NaturalConvectionPOD.mat"
    vPOD = LoadSolutions(root,solname,samples_file,design_space,outputfilename)
    
    solname ="NaturalConvectionValidation/NaturalConvectionValidation"
    samples_file = "NaturalConvectionValidation.txt"
    outputfilename = "NaturalConvectionValidation.mat"
    vValidation= LoadSolutions(root,solname,samples_file,design_space,outputfilename)