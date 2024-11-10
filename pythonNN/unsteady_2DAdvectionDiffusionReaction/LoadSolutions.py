# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:34:17 2020

@author: 56425
"""
import sys
sys.path.insert(0,'./NumSols')


import numpy as np
from Solver import Geometry,BDF,Solver,t0,tn,Nt
from sklearn.decomposition import TruncatedSVD as TruSVD

geo = Geometry()
bdf = BDF(BDForder=3)
K_min = 0.05
K_max = 0.5
design_space =np.array([[K_min,t0],[K_max,tn]])
root='./NumSols/%0.2f_%0.2f'%( design_space[0,0],design_space[1,0] )
NSampleForPOD =10
NSampleForValidation =100

def LoadSolutions(savename,NSample):
    FieldShape   = geo.FieldShape
    Ks = np.linspace(K_min, K_max, NSample)
    tgrid = np.linspace(t0, tn, Nt+1)
    parameters = np.concatenate( (np.tile(Ks[:,None,None],(1, Nt+1, 1)),\
                                  np.tile(tgrid[None,:,None],(NSample, 1, 1)),\
                                  ),\
                                axis = 2\
                                )
    Snapshots = np.zeros((NSample, Nt+1, *FieldShape,1))
    for iK in range(Ks.shape[0]):
        sol = Solver(K=Ks[iK])
        usol = sol.March(bdf)
        for iu in range(len(usol)):
            Snapshots[iK, iu, :,:,0] = usol[iu]
    # save file
    filename = root+"/"+savename
    np.savez_compressed(filename,                 **{"FieldShape":FieldShape,\
                                                     "Snapshots":Snapshots,\
                                                     "tgrid":tgrid,\
                                                     "parameters":parameters,\
                                                     "design_space":design_space})
    return filename

def POD_preprocess(datafile, PODsize=100, Np=30):
    if not datafile.endswith(".npz"):
        datafile = datafile + '.npz'
    datas = np.load(datafile)
    Nt    = datas['tgrid'].size
    if Np > datas['Snapshots'].shape[0]: Np = datas['Snapshots'].shape[0]
    #svd decomposition
    # 1
    init      = datas['Snapshots'][0:1 ,0:1,:,:,:]*0
    # 2
    #init      = datas['Snapshots'].mean(axis=0).mean(axis=0)[None,None,:,:,:]
    snapshots = datas['Snapshots'][0:Np,:  ,:,:,:]-init 
    snapshots  = snapshots[:,:,1:-1,1:-1,:].reshape((Np*Nt,-1,)).T

    max_size = int(5E6)
    if snapshots.size < max_size:
        Modes, sigma, _ = np.linalg.svd(snapshots)
    else:
        svd = TruSVD(n_components=100, n_iter=50, random_state=42)
        Modes = svd.fit_transform(snapshots)
        sigma = svd.singular_values_
    
    
    
    Modes = Modes[:,:PODsize]
    projections = np.matmul(snapshots.T, Modes)
    savename = datafile[:-4]+'_SVD'+str(Np)
    np.savez_compressed(savename,             **{"FieldShape":datas['FieldShape'],\
                                                 "init":init,\
                                                 "Modes":Modes,\
                                                 "sigma":sigma,\
                                                 "tgrid":datas["tgrid"],\
                                                 "projections":projections.reshape((Np,Nt,-1)),\
                                                 "parameters":datas['parameters'][0:Np,:,:],\
                                                 "design_space":datas['design_space']})
    return savename

import matplotlib.pyplot as plt
def showProjs(SVDfile,iM):
    if not SVDfile.endswith(".npz"):
        SVDfile = SVDfile + '.npz'    
    data= np.load(SVDfile)
    projections = data['projections']
    tgrid       = data['tgrid']
    plt.figure()
    plt.plot(tgrid, projections[:,:,iM].T)
    plt.show()

if __name__ == "__main__":
    PODfile        = LoadSolutions("Snapshots_POD", NSampleForPOD)
    SVDfile        = POD_preprocess(PODfile,PODsize=100)
    Validationfile = LoadSolutions("Snapshots_Validation", NSampleForValidation)
    
    for iM in range(10):
        showProjs(SVDfile,iM)
