# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:14:42 2021

@author: Wenqian Chen
"""
import sys,os
sys.path.insert(0,'../../tools')
sys.path.insert(0,'../../tools/NNs')

from Chebyshev import Chebyshev2D
import matplotlib.pyplot as plt    
import numpy as np

"""
eq: K*Hamliton(u)+Vel*Delta(u) + a0 * u =F
bc: u_n = 0

"""
t0 =0
tn =2*np.pi
Nt =200
K  =0.005
a0 =0.01 
BDForder = 3
class Geometry():
    def __init__(self, t0=t0, tn=tn, Nt=Nt, K=K, a0 =a0):
        # problem descprition
        self.FieldShape   = [25, 25]
        self.ndof =np.prod(np.array(self.FieldShape))
        self.t0 = t0
        self.tn = tn
        self.Nt = Nt
        self.dt = (tn-t0)/Nt
        self.K = K
        self.a0 = a0
        
        # spatial discretization
        self.xCoef, self.yCoef = 1/2, 1/2
        self.Chby2D   = Chebyshev2D(xL=-1, xR=1, yD=-1, yU=1, Mx=self.FieldShape[0]-1,My=self.FieldShape[1]-1)
        self.dx, self.dy   = self.Chby2D.DxCoeff(1) 
        self.d2x, self.d2y = self.Chby2D.DxCoeff(2)
        self.dx, self.dy   = self.dx/self.xCoef, self.dy/self.yCoef
        self.d2x, self.d2y   = self.d2x/self.xCoef**2, self.d2y/self.yCoef**2
        self.xc,   self.yc   = self.Chby2D.grid()
        self.xp,   self.yp   = 0.5*self.xc+0.5, 0.5*self.yc+0.5

        # Interior / boundary flag
        self.InteriorShape = (self.FieldShape[0]-2, self.FieldShape[1]-2,)
        self.Interior = np.zeros(self.FieldShape); self.Interior[1:-1,1:-1]=1
        self.Boundary = np.ones(self.FieldShape);  self.Boundary[1:-1,1:-1]=0     
    def Velocity(self,u,t):
        Velx = self.xp*0 + np.cos(t)
        Vely = self.yp*0 + np.sin(t)
        return Velx, Vely
    def Force(self,u,t):
        F = np.exp(-((self.xp-0.5)**2 + (self.yp-0.5)**2)/(0.07**2) )
        return F
    def dForce(self,u,t):
        dF = np.zeros((*self.FieldShape, *self.FieldShape))
        return dF
    def InitField(self,t):
        return self.xc*0
    def Compute_d_dxc(self, phi):
        return np.matmul(self.dx,phi)
    def Compute_d_dyc(self, phi):
        return np.matmul(self.dy, phi.T).T
    def Compute_d_dxc2(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dxc(phi) )
    def Compute_d_dyc2(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dyc(phi) )
    def Compute_d_d1(self, phi):
        return self.Compute_d_dxc(phi), self.Compute_d_dyc(phi)
    def Compute_d_d2(self, phi):
        return self.Compute_d_dxc2(phi), self.Compute_d_dyc2(phi)     
geo = Geometry()

class Solver(Geometry):
    def __init__(self, t0=t0, tn=tn, Nt=Nt, K=K, a0 =a0):
        super(Solver, self).__init__(t0, tn, Nt, K, a0)
    def getJac(self,u,t):
        Velx, Vely = self.Velocity(u,t)
        dF         = self.dForce(u,t)        
        Jac = np.zeros((*self.FieldShape, *self.FieldShape))
        #interior
        for i in range(1,self.FieldShape[0]-1):
            for j in range(1,self.FieldShape[1]-1):
                Jac [i,j,:,j] +=  -self.K * self.d2x[i,:] + Velx[:,j]*self.dx[i,:]
                Jac [i,j,i,:] +=  -self.K * self.d2y[j,:] + Vely[i,:]*self.dy[j,:]
                Jac [i,j,i,j] +=  self.a0 - dF[i,j,i,j]
        return Jac
    def Jac_applyBC(self,Jac,u,t):
        # Dirichlet
        
        # Neumann
        for j in range(0,self.FieldShape[1]):
            Jac [0,j,:,j]  += self.dx[0,:]
            Jac [-1,j,:,j] += self.dx[-1,:]
        for i in range(1,self.FieldShape[0]-1):
            Jac [i,0,i,:]  += self.dy[0,:]
            Jac [i,-1,i,:] += self.dy[-1,:]               
    
    def getRes(self,u,ut,t):
        Velx, Vely = self.Velocity(u,t)
        F          = self.Force(u,t)
        uxc2,uyc2 = self.Compute_d_d2(u)
        uxc ,uyc  = self.Compute_d_d1(u)
        Res = ut -self.K*(uxc2 + uyc2) + Velx*uxc + Vely*uyc + self.a0*u - F
        ResBC = 0*Res
        ResBC [0, :]    = uxc[ 0,:]
        ResBC [-1,:]    = uxc[-1,:]
        ResBC [1:-1,0]  = uyc[1:-1,0]
        ResBC [1:-1,-1] = uyc[1:-1,-1]           
        Res = Res*self.Interior + ResBC*self.Boundary
        return Res
        
    def March(self, BDF, plotflag=False):
        ustates = [self.InitField(self.t0)]        
        usave = [ustates[0]]
        for it in range(self.Nt):
            t = self.t0 + self.dt*(it+1)
            u_pred = BDF.get_upred(ustates)
            ustates.insert(0, u_pred)
            ut = BDF.get_ut(ustates)/self.dt
            Res  =self.getRes(u_pred, ut, t)
            tcoef = BDF.get_tcoeff(ustates)
            
            Jac = self.getJac(u_pred, t)
            self.Jac_applyBC(Jac, u_pred, t)
            Jac = Jac.reshape((self.ndof, self.ndof))  
            
            Jac += np.diag((self.Interior * tcoef/self.dt ).reshape((-1)))
            du = np.linalg.solve(Jac, -Res.reshape((-1,1)))
            
            u_solver = u_pred + du.reshape(self.FieldShape)
            ustates[0] = u_solver
            ut = BDF.get_ut(ustates)/self.dt
            Res  =self.getRes(u_solver, ut, t)
            
            print('Res=%e'%np.linalg.norm(Res.reshape((-1))))
            
            BDF.update(ustates, u_solver)
            if it%10 ==0 and plotflag:
                plt.figure()
                plt.contourf(self.xp, self.yp, ustates[0], np.linspace(-0.1, 0.1, 25))
            usave.append(ustates[0])
        return usave

class BDF():
    def __init__(self, BDForder):
        self.BDForder = BDForder
        self.BDFCoefs = []
        self.BDFExts  = []  
        if BDForder >= 1:
            self.BDFCoefs.append( [1, -1] )
            self.BDFExts.append(  [1]     )
        if BDForder >=2:
            self.BDFCoefs.append([3/2, -2, 1/2])
            self.BDFExts.append( [2 , -1] )
        if BDForder >=3:
            self.BDFCoefs.append( [11/6, -3, 3/2, -1/3] )
            self.BDFExts.append( [3, -3, 1] )
    def get_upred(self,ustates):
        CurrentOrder = len(ustates)
        u_pred = 0
        for i in range(CurrentOrder):
            u_pred += self.BDFExts[CurrentOrder-1][i]*ustates[i]
        return u_pred
    
    def get_ut(self, ustates):
        CurrentOrder = len(ustates)-1
        ut = 0
        for i in range(CurrentOrder + 1):
            ut     += self.BDFCoefs[CurrentOrder-1][i]*ustates[i]
        return ut
    
    def get_tcoeff(self,ustates):
        CurrentOrder = len(ustates)-1
        return self.BDFCoefs[CurrentOrder-1][0]
    
    def update(self, ustates, u):
        CurrentOrder = len(ustates)-1
        ustates[0]=u
        if CurrentOrder == self.BDForder:
            ustates.pop(-1)
            
if __name__ == '__main__':
    plt.close('all')
    sol = Solver()
    bdf = BDF(BDForder)
    usol = sol.March(bdf)
    
    snapshots = np.stack(tuple(usol[::1]), axis=0).reshape((-1,geo.ndof)).T
    Modes, sigma, _ = np.linalg.svd(snapshots)
    plt.figure(); plt.semilogy(sigma); plt.show()