#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:50:49 2020

@author: wenqianchen
"""
import numpy as np

class Chebyshev1D():
    def __init__(self, xL=-1, xR=1, M=5):
        self.xL, self.xR = xL, xR
        self.len = xR-xL
        self.M  = M;
        
    def DxCoeff(self,Ndiff=1):
        M = self.M;
        d=np.zeros((M+1,M+1));
        c=np.ones(M+1);
        c[0]=2;c[M]=2;
        for i in range(M+1):
            for j in range(M+1):
                if i!=j:
                    d[i,j]=c[i]/c[j]*(-1)**(i+j)/2/np.sin((i+j)*np.pi/2/M)/np.sin((j-i)*np.pi/2/M);
                else:
                    d[i,j]=0;
            d[i,i]=-d[i].sum();
        d[0,0]=(2*M*M+1)/6;
        d[M,M]=-d[0,0];
        
        d = d[::-1, ::-1]
        d = d*2/self.len
        dN =d
        
        for i in range(1, Ndiff):
            dN = np.matmul(dN,d);
        return dN
    def grid(self):
        return (self.xL+self.xR)/2 - np.cos( np.arange(self.M+1)*np.pi/self.M ) * self.len/2
    
    # the N-2 iternal points differentation
    def DxCoeffN2(self):
        n =self.M
        x = -np.cos((np.arange(0,n+1)*np.pi/n))
        dx = np.zeros((n+1,n+1))
        for j in range(1,n):
            for i in range(1,n):
                if i == j:
                    dx[j,j] = 1.5*x[j]/(1-x[j]**2)
                else:
                    dx[j,i] = (-1)**(i+j)*(1-x[i]**2)/((1-x[j]**2)*(x[j]-x[i]))
        return dx
                    
    
class Chebyshev2D():
    def __init__(self, xL=-1, xR=1, yD=-1, yU=1, Mx=5,My=5):
        self.xL, self.xR, self.yD,self.yU = xL,xR,yD,yU
        self.xlen = xR-xL
        self.ylen = yU-yD
        self.Mx  = Mx
        self.My  = My
        self.xChby = Chebyshev1D(xL=self.xL, xR=self.xR, M=self.Mx)
        self.yChby = Chebyshev1D(xL=self.yD, xR=self.yU, M=self.My)
        
    def DxCoeff(self,Ndiff=1):
        dxN = self.xChby.DxCoeff(Ndiff)
        dyN = self.yChby.DxCoeff(Ndiff)
        return dxN, dyN
    def DxCoeffN2(self):
        dx = self.xChby.DxCoeffN2()
        dy = self.yChby.DxCoeffN2()
        return dx, dy   

    def grid(self):
        y,x = np.meshgrid(self.yChby.grid(), self.xChby.grid())
        return x,y
    
# unit test        
if __name__ == "__main__":
    M = 10;
    xL = 3; xR=5;
    Cheb1D = Chebyshev1D(xL, xR,M);
    xgrid  = Cheb1D.grid()[:,None];
    dx     = Cheb1D.DxCoeff();
    d2x    = Cheb1D.DxCoeff(2);
    dxp    = Cheb1D.DxCoeffN2()
    
    dx_xgrid = np.matmul(dx,xgrid);
    d2x_xgrid= np.matmul(d2x,xgrid**2);
    dxp_xgrid= np.matmul(dxp,xgrid**2);
    
    print('dx:',dx_xgrid)
    print('d2x:',d2x_xgrid)
    print('dxp:',dxp_xgrid)
    
    Cheb2D=Chebyshev2D(xL=-1, xR=1, yD=-1, yU=1, Mx=5,My=5)