# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:34:17 2020

@author: 56425
"""

import struct
import numpy as np
import pandas as pd
from NaturalConvection import Geometry
from sklearn.decomposition import TruncatedSVD as TruSVD

geo = Geometry()
FieldShape   = geo.FieldShape

samples_file = "NaturalConvection.txt"
solname      = "NaturalConvection"


def LoadSolutions(root,design_space,Nt,savename, NSample0, NSamplet):
    tgrid = np.linspace(design_space[0,-1], design_space[1,-1],Nt)[:,None]
    df = pd.read_csv(root+"/"+samples_file,skiprows=6,header=None, sep="\t");
    alpha_p = df.values[NSample0-1:NSamplet-1,1:3]
    alpha_p = np.tile(alpha_p[:,None,:], (1, Nt, 1))
    t       = np.tile(tgrid[None,:,:], (NSamplet-NSample0,1,1))
    parameters = np.concatenate((alpha_p, t), axis=2)
    Snapshots = np.zeros((0, Nt, *FieldShape,4))
    for i in range(NSample0, NSamplet):
        print('processing %dth sample'%(i,))
        tmp = np.zeros((0, *FieldShape,4))
        for nt in range(tgrid.shape[0]):
            import os
            print(i,nt)
            samplesolfile = root+os.sep+solname+os.sep+solname+ "_%d/OUTPUT/Time=%.3f/RESULT.plt"%(i, tgrid[nt]);
            snapshot = loadsnapshot(samplesolfile)
            tmp = np.append(tmp, snapshot[None,:,:,:], axis=0)
        Snapshots = np.append(Snapshots, tmp[None,:,:,:,:], axis=0)
    # save file
    filename = root+"/"+savename
    np.savez_compressed(filename,                 **{"FieldShape":FieldShape,\
                                                     "Snapshots":Snapshots,\
                                                     "tgrid":tgrid,\
                                                     "parameters":parameters,\
                                                     "design_space":design_space})
    return filename
def loadsnapshot(file):
    with open(file,'rb') as f:
        FieldShape_read = np.array(struct.unpack('ii',f.read(8)))+1
        if np.any(FieldShape_read != np.array(FieldShape)):
            raise Exception('unmatched field shape')
        f.read(24)
        size = FieldShape[0]*FieldShape[1]*4
        snapshot=np.array(struct.unpack('%dd'%size,f.read(8*size)))\
                 .reshape((FieldShape[1],FieldShape[0],4))\
                 .transpose((1,0,2));
        # define the center as reference point for pressure
        #snapshot[:,:,0] = snapshot[:,:,0]/BetaP
        snapshot[:,:,0]=snapshot[:,:,0] - snapshot[(FieldShape[0]-1)//2, (FieldShape[1]-1)//2, 0] 
    return snapshot[::-1, ::-1,:]

def POD_preprocess(datafile, PODsize=100, Np=30):
    if not datafile.endswith(".npz"):
        datafile = datafile + '.npz'
    datas = np.load(datafile)
    Nt    = datas['tgrid'].size
    if Np > datas['Snapshots'].shape[0]: Np = datas['Snapshots'].shape[0]
    #svd decomposition
    # 1
    #init      = datas['Snapshots'][0:1 ,0:1,:,:,:]
    # 2
    init      = datas['Snapshots'].mean(axis=0).mean(axis=0)[None,None,:,:,:]
    # 3
    #init      = datas['Snapshots'][0:1 ,0:1,:,:,:]*0
    #init[0:1 ,0:1,0,:,3]=0.5; init[0:1 ,0:1,-1,:,3]=-0.5
    snapshots = datas['Snapshots'][0:Np,:  ,:,:,:]-init 
    TBC       = geo.TBC[None,None,:,:] * geo.Tfun(datas['tgrid'][None,:,:,None], sin=np.sin)
    snapshots[:,:,:,:,3] = snapshots[:,:,:,:,3] - TBC
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
                                                 "projections":projections,\
                                                 "parameters":datas['parameters'][0:Np,:,:],\
                                                 "design_space":datas['design_space']})

import matplotlib.pyplot as plt
def showSnapshot(datafile, ip=0, it=0, Nt=0):
    if not datafile.endswith(".npz"):
        datafile = datafile + '.npz'
    datas = np.load(datafile)
    snapshot = datas['Snapshots'][ip,it,:,:,:]
    alpha    = datas['parameters'][ip,it,:][None,:]
    xp,yp=geo.getGrid(alpha)
    fig =plt.figure()
    gs = fig.add_gridspec(2, 2)
    fig.add_subplot( gs[0, 0] ).contourf(xp,yp,snapshot[:,:,0])
    fig.add_subplot( gs[0, 1] ).contourf(xp,yp,snapshot[:,:,1])
    fig.add_subplot( gs[1, 0] ).contourf(xp,yp,snapshot[:,:,2])
    fig.add_subplot( gs[1, 1] ).contourf(xp,yp,snapshot[:,:,3])
    plt.show()

def showNu(datafile, ip, ivar=0):
   if not datafile.endswith(".npz"):
       datafile = datafile + '.npz'
   datas = np.load(datafile) 
   tgrid = datas['tgrid']
   rec = datas['Snapshots'][ip,:, 2:16,16,ivar]
#   rec = np.zeros((2,0))
#   for it in range(Nt):
#       ind = ip*Nt+it
#       uv = datas['Snapshots'][ind,16,16,2:4]
#       rec = np.append(rec,uv[:,None],axis=1)
   plt.figure()
   plt.plot(tgrid,rec,'*-')
   plt.show()
if __name__ == "__main__":
    design_space = np.array([[1E4,75,0],[3E4,90,100]])   
    Nt= 1000+1
    root='../NumSols/%0.0E_%0.0Eand%0.1d_%0.1d'%(design_space[0,0],design_space[1,0], \
                                              design_space[0,1],design_space[1,1], \
                                              )
    #root = root + "__refine"
    #root = 'test_No_ACM'
    #savename = "Snapshots_POD"
    NSample0ForPOD = 1; NSampletForPOD=60+1
    #PODfile        = LoadSolutions(root,design_space,Nt,"PODG_Snapshots_POD"       ,NSample0ForPOD, NSampletForPOD)

    PODfile = "../NumSols/1E+04_3E+04and75_90/PODG_Snapshots_POD.npz"
    POD_preprocess(PODfile,PODsize=60,Np=60)
    
    #Validationfile = LoadSolutions(root,design_space,Nt,"PODG_Snapshots_Validation",NSampletForPOD, 101)
    
    
#    for it in range(11):
#        showSnapshot(PODfile, ip=1, it=it, Nt=Nt)
#    for i in range(10):
#        showNu(PODfile, ip=i,ivar=0)