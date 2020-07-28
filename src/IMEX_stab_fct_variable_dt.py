#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:51:03 2020

@author: jf
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import fcs as fcs
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
#####################################################
##################### Setup #########################
#####################################################

a = 0
b = 1
N=5

lf= 2
ls = 1


dt = 0.2




def F(t,lf=lf,ls = ls):
    return np.exp(1j*(lf+ls)*t)

def ff(t,y,l=lf):
    return 1j*l*y

def fs(t,y, l=ls):
    return 1j*l*y

def f(t,y, ff=ff, fs = fs): 
    return ff(t,y) + fs(t,y)


y0 = F(a)
nodes,weights = fcs.rnw(N, a, b) #FCS.nodes_weights
t = (1-nodes)[::-1]
t = np.append(0,t)
t = np.append(t,b)

def IMEX_matrix(lf,ls,t):
    import numpy as np
    dim = len(t)-2
    diff = np.diff(t[0:-1])
    R = np.zeros([dim,dim],dtype=np.complex)
    terms = (1+1j*diff*ls)/(1-1j*diff*lf)
    diag = np.cumprod(terms)
    np.fill_diagonal(R,diag)
    return R

mat = IMEX_matrix(lf, ls, t)
res = np.zeros(N,dtype=np.complex)
res[:]= y0


def IMEX_c(lf,ls,y0,t):
    import numpy as np
    n = len(t)
    y = np.zeros(n,dtype=np.complex)
    y[0] = y0
    dt_v = np.diff(t)
    for i in range(1,n):
        dt = dt_v[i-1]
        y_ex = y[i-1] + dt*1j*ls*y[i-1] # explicit part 
        y[i] = y_ex/(1-1j*dt*lf) #implicit part
    return y

y_result = IMEX_c(lf,ls,y0,t)
res = mat@res #result is same as IMEX_c 

fig, ax = plt.subplots()
ax.plot(t,F(t).real,label=r'Re$(F(x))$')
ax.plot(t,F(t).imag,label=r'Im$(F(x))$')
ax.plot(t,y_result.real,label=r'Re$(y)$')
ax.plot(t,y_result.imag,label=r'Im$(y)$')
ax.plot(t[1:-1],res,linestyle='--')
ax.plot(t[1:-1],res.imag,linestyle='--')

ax.legend()