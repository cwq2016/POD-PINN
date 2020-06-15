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
    parameters = df.values[:,1:3]
    Nsample = parameters.shape[0]
    Samples = []
    k=0
    
    for i in range(Nsample):
        samplesolfile = solroot+"/"+solname+ "_%d/OUTPUT/Time=0.100/RESULT.plt"%(i+1);
        if os.path.isfile(samplesolfile):
            with open(samplesolfile,'r') as f:
                line = f.readline();
                line = f.readline();
                line = f.readline();
        else:
            print(i+1)
            k=k+1
            continue
        FieldShape =np.array([ int(num) for num in re.findall(r"\d+",line)])
        df = pd.read_csv(samplesolfile,skiprows=3,header=None, sep="\s+", usecols=[0,1,2,3,4,5,6,7]);
        snapshot = np.reshape( df.values, (*FieldShape, 8, ), "F")
        # define the center as reference point for pressure
        snapshot[:,:,2]=snapshot[:,:,2] - snapshot[(FieldShape[0]-1)//2, (FieldShape[1]-1)//2 , 2] 
        Samples.append( snapshot[::-1,::-1,2:].reshape((-1), ))
    Samples = np.stack(tuple(Samples), axis=1)
    print(k)
    savemat(solroot+"/"+outputfilename, {"FieldShape":FieldShape,\
                                          "Samples":Samples,\
                                          "parameters":parameters,\
                                          "design_space":design_space})
if __name__ == "__main__":
    design_space = np.array([[100,60],[500,120]])
    root='%d_%dand%d_%d'%(design_space[0,0],design_space[1,0], \
                          design_space[0,1],design_space[1,1])
    
    solname ="LidDrivenPOD/LidDrivenPOD"
    samples_file = "LidDrivenPOD.txt"
    outputfilename = "LidDrivenPOD.mat"
    LoadSolutions(root,solname,samples_file,design_space,outputfilename)
    
    solname ="LidDrivenValidation/LidDrivenValidation"
    samples_file = "LidDrivenValidation.txt"
    outputfilename = "LidDrivenValidation.mat"
    LoadSolutions(root,solname,samples_file,design_space,outputfilename)