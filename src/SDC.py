#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:16:02 2020

Example for SDC using Euler explicit as integrator

@author: jf
"""
import numpy as np
from functions import fcs as fcs
import matplotlib.pyplot as plt

#####################################################
##################### Setup #########################
#####################################################

a = -1.5
b = 1.5
N = 8 #number of grid points between a and b
SDC_iterations=1
equispaced= True # equispaced grid or GL nodes

#####################################################
######## Definition of analytic sol and RHS #########
#####################################################

def F(t):
    #return 3.2*t**7 + 4.1*t**4 + 9.2*t**2 
    return np.exp(-t)
    #return np.sin(t)

def f(t,x): 
    #return 7*3.2*t**6 + 4*4.1*t**3 + 2*9.2*t**1 
    return -x
    #return np.cos(t)


#####################################################
############## Creating grid points  ################
#####################################################
    
if equispaced:
    t = [a,b]  
    t = fcs.refine_grid(t,N)
else:
    nodes, weights = fcs.GL(N,a,b) # weights are not need
    t = np.append(a,nodes) # create t_i including endpoints
    t = np.append(t,b)

####################################################
################# Integrating ######################
####################################################

t,y = fcs.SDC(f,a,b,F(a),t,iterations=SDC_iterations)


####################################################
############### Calculate Polynomial ###############
####################################################

# calculates lagrange polynomial using newton basis polynomials
tt, yy = fcs.Newton_polynomial(t, y,100) # just for plotting


####################################################
################# Visualization ####################
####################################################

#time series to plot analytical solution
x = np.linspace(a,b,100)


fs = 11 #fontsize

# setup plot for .pgf output and LaTeX integration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'CronosPro-Regular' 
plt.rcParams.update({'font.size': fs})
     
# Create figure   
fig1 = plt.figure(figsize=(5.6,4),frameon=True)
ax1 = fig1.add_subplot(111)

# plot graphs
ax1.plot(x,F(x),label=r'$\varphi(t)$',color='C0',lw=1,linestyle='--',dashes=(7, 1))
ax1.plot(tt,yy,label=r'$\varphi_c(t)$',color='C3',lw=1,linestyle=':',dashes=(5, 3))
ax1.plot(t,y,label=r'$\varphi_c(t_i)$',linestyle='',marker='.',markersize=10,color='C3')
      
# some visuals 
ax1.tick_params(axis="y",direction="in")
ax1.tick_params(axis="x",direction="in")
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')


ax1.legend(fancybox=True,loc='best')
#ax1.set_xlim(-0.05+a,b+0.05) # 0,40
#ax1.set_ylim(-15,100)
ax1.set_xlabel('t')
#ax1.set_ylabel(ylabel)
ax1.set_title('SDC, iterations='+str(SDC_iterations),fontsize=fs)
fig1.tight_layout()