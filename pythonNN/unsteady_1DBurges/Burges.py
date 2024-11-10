#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:53:35 2020

solving 1DBurges problem:
    u_t + u*u_x = 1/Re * u_xx
    u(x,0)=u0(x), u(0,t)=u(L,t)=0
    where u0(x) = [ x/(t+1) ] / [1+sqrt((t+1)/t0)*exp(Re*x^2/(4*t+4))]
    t0 = exp(Re/8)
    ## POD 
    u = V*lamda + u0
    # ROM
    (V*lamda+u0)_t + (u_V*lamda+u0)*(V*lamda+u0)_x = 1/Re * (V*lamda+u0)_xx
    # reduced cost
    lamda_t = lamd^T*A*lamda + (B+Binit)*lamda + source 
    
@date: 07/13/2020
@author: wenqianchen
"""

import sys
sys.path.insert(0,'../tools')
sys.path.insert(0,'../tools/NNs')

from Chebyshev import Chebyshev1D
import numpy as np
import torch
from Normalization import Normalization

# reproducible
#torch.manual_seed(1234)  
#np.random.seed(1234)


class CustomedEqs():    
    def __init__(self,NSample=10, tend=2, xlen=1, Nt=100, Re_min = 200, Re_max=800):
        self.NSample    = NSample
        self.Re         = np.linspace(Re_min,Re_max,self.NSample+1)[:,None]
        
        self.Nt      = Nt
        self.tend    = tend
        self.tgrid   = np.linspace(0,self.tend,self.Nt+1)[:,None]
        
        
        Re_t_grid = np.meshgrid(self.Re, self.tgrid)
        self.parameters = np.stack(Re_t_grid, axis = 2).transpose((1,0,2)).reshape((-1,2))
        
        self.Np = 128
        # spatial discretization
        Cheb1D  = Chebyshev1D(0, xlen, self.Np)
        self.xgrid = Cheb1D.grid();
        self.dx      = Cheb1D.DxCoeff();
        self.d2x     = Cheb1D.DxCoeff(2);
        

        
        self.design_space = np.array([[Re_min,0],[Re_max,self.tend]])
        #generate snapshots with exact solution     
        self.Snapshots = self.ExactSolution(self.parameters).T-\
                         self.InitialSolution(self.parameters[:,0:1]).T
        # svd decomposition
        self.AllModes, self.sigma, _ = np.linalg.svd(self.Snapshots);

    def setM(self,M):
        self.Modes = self.AllModes[:,:M]
        self.M = M
        
        self.projections = np.matmul( self.Modes.T, self.Snapshots)
        _, Mapping  = Normalization.Mapstatic(self.projections.T)
        self.proj_mean =  Mapping[0][None,:] 
        self.proj_std  =  Mapping[1][None,:] 
         
        
        
        self.Modes_x  = np.matmul(self.dx, self.Modes)
        self.Modes_x[0,:]=0; self.Modes_x[-1,:]=0    # 0-value boundary condition
        self.Modes_xx = np.matmul(self.d2x, self.Modes)
        self.Modes_xx[0,:]=0; self.Modes_xx[-1,:]=0    # 0-value boundary condition
        
        ## Reduced-order model constants
        self.A = self.getA()
        self.B = self.getB()
        
        
    def ExactSolution(self,alpha):
        Re = alpha[:,0:1]
        t  = alpha[:,1:2]
        x  = self.xgrid.reshape((1,-1)) 
        t0= np.sqrt(Re/8)
        u = ( x/(t+1) ) / (1+np.sqrt((t+1)/t0)*np.exp(Re*x**2/(4*t+4)))
        return u   
    def InitialSolution(self,Re):
        return self.ExactSolution( np.hstack((Re, np.zeros_like(Re))) )

    # get A from the first mth modes
    def getA(self): 
        A = np.matmul( self.Modes.reshape((-1,self.M,1)), self.Modes_x.reshape(-1,1,self.M))
        A = A[None,:]*self.Modes.T.reshape((self.M, -1,1,1))
        A = A.sum(axis=1).squeeze().reshape((self.M, self.M, self.M))
        return A        
    
        
    def getB(self):
        B = np.matmul(self.Modes.T,-self.Modes_xx)
        return B
        
    def getBinit(self,Re):
        N =Re.shape[0]
        u0 =self.InitialSolution(Re) #N*Np
        u0_x = np.matmul(u0, self.dx.T)       #N*Np
        B = np.zeros((N,self.M, self.M))            #M*M
        for k in range(N):
            for i in range(self.M):
                for j in range(self.M):
                    # u0*du_x
                    B0   = np.sum(         u0[k,:] * self.Modes[:,i] * self.Modes_x[:,j] )
                    # u*du0_x
                    B1   = np.sum( self.Modes[:,j] * self.Modes[:,i] *      u0_x[k,:]    )
                    B[k,i,j] = B0 +  B1
        return B
    def getsource(self, Re):
        N =Re.shape[0]
        u0 =self.InitialSolution( Re) #N*Np
        u0_x = np.matmul(u0, self.dx.T )      #N*Np
        u0_xx= np.matmul(u0, self.d2x.T)      #N*Np
        source = np.zeros((N,self.M))
        for k in range(N):
            for i in range(self.M):
                s0 =  np.sum(self.Modes[:,i]*u0[k,:]*u0_x[k,:])
                s1 = -1/Re[k,0]*np.sum(self.Modes[:,i]*u0_xx[k,:])
                source[k,i] = s0+s1
        return source
    
    
    def rhs(self,lamda,t,Re):
        lamda = lamda.reshape(-1, self.M)
        fx   = np.matmul(lamda[:,None,None,:], self.A[None,:,:,:])
        fx   = np.matmul(fx,lamda[:,None,:,None])
        fx   = fx.reshape(lamda.shape)
        Binit    = self.getBinit(Re).squeeze()
        source = self.getsource(Re)
        fx   = fx +  np.matmul( lamda, 1/Re *self.B.T + Binit.T) +source
        fx   = -fx
        return fx.squeeze()
    
    def POD_G(self,Re):
        n = Re.shape[0]
        lamda  = np.zeros((n, self.Nt+1,self.M))
        from scipy.integrate import odeint
        for i in range(n):
            u0 = np.zeros((self.M))
            lamda[i,:,:]=odeint(lambda f, t: self.rhs(f,t,Re[i:i+1,:]), u0, self.tgrid.flatten())
        u_PODG = np.matmul(lamda,self.Modes.T)+self.InitialSolution(Re)[:,None,:]
        return lamda,u_PODG
    
    def GetPODGError(self, Re):
        lamda_PODG,_ = self.POD_G(Re)
        Error_PODG = self.GetglobalError(Re,lamda_PODG)
        return Error_PODG
    
    def getlocalError(self, alpha, lamda):
        Re = alpha[:,0:1]
        u_pred         = np.matmul( lamda, self.Modes.T) + self.InitialSolution(Re)
        u_Exact        = self.ExactSolution(alpha)
        Error = np.linalg.norm(u_Exact-u_pred, axis = 1)
        return Error
    
    def GetglobalError(self,Re,lamda):
        alpha = np.stack(np.meshgrid(Re,self.tgrid), axis = 2).transpose((1,0,2)).reshape((-1,2))
        lamda = lamda.reshape((-1, self.M))
        Error = self.getlocalError(alpha, lamda).reshape((-1, self.Nt+1))
        weight= np.ones_like(Error)
        weight[0,:]=0.5; weight[-1,:]=0.5;
        Error = np.sqrt( (weight * Error**2).sum(axis=1) )[:,None]
        return Error.mean()
    
    def GetProjError(self, Re):
        alpha       = np.stack(np.meshgrid(Re,self.tgrid), axis = 2).transpose((1,0,2)).reshape((-1,2))
        du_Exact    = self.ExactSolution(alpha) - self.InitialSolution(alpha[:,0:1])
        lamda_Exact= np.matmul( du_Exact, self.Modes ).reshape((-1, self.Nt+1, self.M))
        return self.GetglobalError(Re,lamda_Exact) 

from NN import GeneralNet,DEVICE   
class CustomedNet(GeneralNet):
    def __init__(self, layers=None,oldnetfile=None,roeqs=None, WeightLog=0):
        super(CustomedNet, self).__init__(layers=layers,OldNetfile=oldnetfile)
        self.M = torch.tensor(roeqs.M).long().to(DEVICE)
        self.A = torch.tensor( roeqs.A).float().to(DEVICE)
        self.B = torch.tensor( roeqs.B ).float().to(DEVICE)
        self.lb= torch.tensor(roeqs.design_space[0:1,:]).float().to(DEVICE)
        self.ub= torch.tensor(roeqs.design_space[1:2,:]).float().to(DEVICE)
        self.roeqs = roeqs
        self.proj_std = torch.tensor( roeqs.proj_std ).float().to(DEVICE)
        self.proj_mean= torch.tensor( roeqs.proj_mean).float().to(DEVICE)
        self.WeightLog=WeightLog
        
    def u_net0(self,x):
        x = (x-(self.ub+self.lb)/2)/(self.ub-self.lb)*2
        Re = x[:,0:1]
        t  = x[:,1:2]
        x0 = torch.cat((Re,-torch.ones_like(t)),1)
        out = self.unet(x)-self.unet(x0)
        out = out*self.proj_std
        return out

    def u_net(self,x):
        x = (x-(self.ub+self.lb)/2)/(self.ub-self.lb)*2
        out = self.unet(x)
        out = out*self.proj_std + self.proj_mean
        return out
    
    def forward(self,x):
        return self.u_net(x).detach().cpu().numpy()
    
    def loss_NN(self, xlabel, ylabel):
        y_pred    = self.u_net(xlabel)
        diff      = (ylabel-y_pred)/self.proj_std
        loss_NN   = self.lossfun(diff,torch.zeros_like(diff))
        return loss_NN
    
    def loss_PINN(self,x,Binit,source,weight=1):
        Re  = x[:,0:1]
        t   = x[:,1:2]
        t.requires_grad_(True)
        lamda = self.u_net( torch.cat( (Re,t),1 ) )
        dlamda=torch.zeros_like(lamda)
        for i in range(self.M):
            dlamda[:,i:i+1] = self.grad(lamda[:,i:i+1].sum(), t)   
            
        fx   = torch.matmul(lamda[:,None,None,:], self.A[None,:,:,:])
        fx   = torch.matmul(fx,lamda[:,None,:,None])
        fx   = fx.view(lamda.shape)
        fx   = fx + 1/Re * torch.matmul( lamda, self.B.T) + torch.matmul(Binit, lamda[:,:,None]).view(lamda.shape)
        resi = dlamda + fx + source
        #resi = resi/self.proj_std
        return self.lossfun(weight*resi,torch.zeros_like(fx))
    def loss_Hybrid(self, data_label, data_resi):
        return self.loss_NN(*data_label) +(10**self.WeightLog)* self.loss_PINN(*data_resi)
    
if __name__ == '__main__':
    M = 10
    NSample=20; tend=2; xlen=1; Nt=100; Re_min = 200; Re_max=800
    roeqs = CustomedEqs(NSample=NSample, tend=tend, xlen=xlen, Nt=Nt, Re_min = Re_min, Re_max=Re_max)
    roeqs.setM(M)
    
#    Net = CustomedNet(roeqs=roeqs,layers=[2,20,20,20,M])
#    print(Net.labeledLoss)
        
    from plotting import newfig,savefig
    import matplotlib.pyplot as plt    
    showSingularValue = False
    if showSingularValue:
        newfig(width=0.8)
        plt.semilogy(np.arange(roeqs.sigma.shape[0])+1, roeqs.sigma,'-ko')
        plt.xlabel('$m$')
        plt.ylabel('Singular value')    
        plt.show()
        savefig('fig/SingularValues_%d'%(roeqs.NSample) )
    #showSingularValue(roeqs)
    showModes = False 
    if showModes:
        Nrow = 2
        Ncol = 2
        import matplotlib.gridspec as gridspec
        for i in range(roeqs.M//(Nrow*Ncol)+1):
            iM0 = i*Nrow*Ncol
            if iM0<roeqs.M:
                fig =newfig(width=1.5,nplots=Nrow/Ncol)
                gs = fig.add_gridspec(Nrow, Ncol)
                plt.show()
            for irow in range(Nrow):
                for icol in range(Ncol):
                    iM = iM0 + irow*Ncol + icol
                    print('iM=%d\n'%iM, irow, icol)
                    ax = fig.add_subplot( gs[irow, icol] )
                    if iM<roeqs.M:
                        ax.plot(roeqs.xgrid,roeqs.Modes[:,iM])
                        #ax.set_title('m=%d'%iM)
    def show(Re): 
        for i in range(Re.shape[0]):
            newfig(width=1)
            interval = 10
            alpha = np.stack(np.meshgrid(Re[i:i+1,:],roeqs.tgrid[::interval,:]), axis = 2).transpose((1,0,2)).reshape((-1,2))
            uE = roeqs.ExactSolution(alpha)
            _, uPOD_G = roeqs.POD_G(Re[i:i+1,:])
            uPOD_G   = uPOD_G.squeeze(axis=0)[::interval,:]
            plt.plot(roeqs.xgrid, uE.T,'r-', roeqs.xgrid, uPOD_G.T,'g--')
            plt.title('Re=%f'%Re[i:i+1,:])
            plt.show()
        
    debugshow = False
    if debugshow:
        show(np.linspace(Re_min,Re_max,3)[:,None])

    # projection error
    Re= roeqs.Re
    Error = roeqs.GetProjError(Re)
    Error_PODG = roeqs.GetPODGError(Re)
    
    print(np.hstack((Error, Error_PODG)))
