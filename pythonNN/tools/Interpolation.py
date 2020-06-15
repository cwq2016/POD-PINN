# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:01:11 2020

@author: 56425
"""

import numpy as np

# constuct interpolation matrix for 1D points
class Interpolation():
    def __init__(self, xin, xout):
        pass
    @staticmethod
    def BarycentricWeights(xin):
        Nin = xin.shape[0]
        wBarry = np.ones_like(xin)
        for i in range(1,Nin+1):
            for j in range(i):
                wBarry(j) *= xin(j)-xin(i) 
                wBarry(i) *= xin(i)-xin(j)
                
        return 1/wBarry
    @staticmethod
    def LagrangeInterpolationPolys(xin, xout, xinBarry):
        Nin  = xin.shape[0]
        Nout = xout.shape[0]
        L = np.zeros((Nout, Nin))
        for iout in range(Nout):
            xi = xout[i]
            xi_in_xin = abs(xi-xin)<1E-14
            if any(xi_in_xin):
                L[iout, xi_in_xin] = 1.0
                break;
            L[iout,:] = L[iout,:]/(xi - xinBarry)
            L[iout,:] = L[iout,:]/L[iout,:].sum()
        return L
            
            