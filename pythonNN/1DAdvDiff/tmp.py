#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:00:33 2020

@author: wenqianchen
"""
from sympy import diff, symbols,exp
a=1
x,y,z= symbols('x y z')
FUNC=x**2+y**2+a*exp(z)
dx=diff(FUNC,x)
dy=diff(FUNC,y)
dz=diff(FUNC,z)

y = lambdify(((x, y, z),), dx)