#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original Equations:
    > V*phi_x - a*phi_xx = f
    > where V=a=1
    * the artifical solutions are defined as 
    > phi = exp(-alpha2*x).*(1+alpha1*x)*(x^2-1)
    > where alpha = (alpha1, alpha2) in [1, 10]x[1,10] is design parameters
Reduced order equations:
    > phi_Modes‘*(V*Dx-a*D2x)*phi_Modes*lamda = phi_Modes'*f
    


Created on Wed Mar 18 14:40:38 2020

@author: wenqianchen
"""



import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NN')

from  ReducedOrderEqs import ReducedOrderEquationns as ROEqs
from Chebyshev import Chebyshev1D
from scipy.io import loadmat
import numpy as np
import torch
import torch.autograd as ag
from NN import POD_Net, train, DEVICE
from Normalization import Normalization

V  = 1
a  = 1


torch.manual_seed(12)  # reproducible
np.random.seed(1234)
#DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Eqs1D(ROEqs):    
    def __init__(self, matfile, M):
        datas = loadmat(matfile)
        self.Samples = datas['Samples']
        self.xgrid   = datas['xgrid']
        self.parameters = datas['parameters']
        self.design_space = datas['design_space']
        
        
        self.Np      = self.Samples.shape[0]-1
        self.NSample = self.Samples.shape[1]
        
        # svd decomposition
        self.Modes, self.sigma, _ = np.linalg.svd(self.Samples);
        self.Modes = self.Modes[:,:M]
        self.M = M
        
        
        
        # spatial discretization
        Cheb1D  = Chebyshev1D(self.xgrid[0], self.xgrid[-1], self.Np)
        self.dx      = Cheb1D.DxCoeff();
        self.d2x     = Cheb1D.DxCoeff(2);
        self.projections = np.matmul( self.Modes.T, self.Samples)
        _, Mapping  = Normalization.Mapstatic(self.projections.T)
        self.proj_mean =  Mapping[0][None,:] 
        self.proj_std  =  Mapping[1][None,:] 
        _, self.Binv = self.getB()
        
    # get A from the first mth modes
    def getA(self):
        A = np.zeros((self.M, self.M, self.M))
        return A        
        
    def getB(self):
        tmp =V*self.dx-a*self.d2x
        # add boundary conditions
        tmp[0,:] =0; tmp[-1, :]=0;
        tmp[0,0] =1; tmp[-1,-1]=1;
        
        tmp = np.matmul(self.Modes.T, tmp)
        B = np.matmul(tmp,self.Modes)
        Binv = np.linalg.inv(B)
        return B, Binv
    
    # Normalization, x= xh*sigma+miu
    def getBNormal(self):
        B, Binv = self.getB()
        B = B*self.proj_std
        Binv = Binv/self.proj_std.T
        return B, Binv

    def POD_G(self,Mchoose, alpha):
        alpha1 = alpha[:,0:1]
        alpha2 = alpha[:,1:2]
        source = self.getsource(alpha1, alpha2)
        lamda = np.matmul(source, self.Binv.T)
        return lamda
    
    def GetError(self,alpha,lamda):
        alpha1 = alpha[:,0:1]
        alpha2 = alpha[:,1:2]
        phi_pred         = np.matmul( lamda, self.Modes.T)
        phi_Exact        = self.phix(self.xgrid.T, alpha1, alpha2)
        Error = np.linalg.norm(phi_Exact-phi_pred, axis = 1)/np.linalg.norm(phi_Exact, axis=1)
        Error = Error[None,:]
        Error = Error.mean()
        return Error
        
    def getsource(self,alpha1, alpha2, exp=np.exp):
        x = self.xgrid.T
        f1    = (1+alpha1*x)*(x**2-1)
        f2    = exp(-alpha2*x/3)
        f1_x  = 3*alpha1*x**2 +2*x -alpha1
        f2_x  = -alpha2/3*exp(-alpha2*x/3)
        f1_xx = 6*alpha1*x + 2
        f2_xx = alpha2**2/9*exp(-alpha2*x/3)
        
        phi_x  = f1*f2_x + f1_x*f2
        phi_xx = 2*f1_x*f2_x + f1*f2_xx +f1_xx*f2
        source = V*phi_x - a*phi_xx
        source[:, 0:1] =self.phix(x[0,  0], alpha1, alpha2)
        source[:,-1: ] =self.phix(x[0, -1], alpha1, alpha2)
        source = np.matmul( source, self.Modes )
        return source 

    # Normalization, x= xh*sigma+miu        
    def getsourceNormal(self, alpha1, alpha2, exp=np.exp):
        source = self.getsource(alpha1, alpha2, exp=np.exp)
        B, _ = self.getB()
        source = source - np.matmul(self.proj_mean, B.T)
        return source
    
    def phix(self,x,alpha1,alpha2, exp = np.exp):
        return exp(-alpha2*x/3)*(1+alpha1*x)*(x**2-1)
    

        
        
    
class Net1DAdvDiff(POD_Net):
    def __init__(self, layers=None,oldnetfile=None,roeqs=None):
        super(Net1DAdvDiff, self).__init__(layers=layers,OldNetfile=oldnetfile)
        self.M = roeqs.M
        self.A = torch.tensor( roeqs.getA() ).float().to(DEVICE)
        B, Binv = roeqs.getBNormal()
        self.B = torch.tensor( B ).float().to(DEVICE)
        self.Binv = torch.tensor( Binv ).float().to(DEVICE)
        self.lb = torch.tensor(roeqs.design_space[0:1,:]).float().to(DEVICE)
        self.ub = torch.tensor(roeqs.design_space[1:2,:]).float().to(DEVICE)
        self.roeqs = roeqs


    def u_net(self,x):
        x = (x-(self.ub+self.lb)/2)/(self.ub-self.lb)*2
        out = self.unet(x)
        #out = out*self.out_std + self.out_mean
        return out
    
    def forward(self,x):
        return self.u_net(x).detach().cpu().numpy()*self.roeqs.proj_std + self.roeqs.proj_mean
    
    def loss_PINN(self,x,source):
        lamda = self.u_net(x);
        fx = torch.matmul( lamda, self.B.T) -source
        fx = torch.matmul( fx, self.Binv.T)
        return self.lossfun(fx,torch.zeros_like(fx))
        
    def grad(self,a,b):
        if b.grad is not None:
            b.grad.zero_()
        da_db = ag.grad(a, b, None, 
                   create_graph=True, retain_graph=True)[0]
        return da_db
        
        
        
        