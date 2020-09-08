#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:48:01 2020

@author: jf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 300
c_s = 0.2
U = 0.1
x1 = 0.25
x0 = 0.75
sigma = 0.1
k = 7.2*np.pi
S = np.asmatrix([[1,1],[1,-1]])
S_1 = np.asmatrix([[0.5,0.5],[0.5,-0.5]])
A = np.asmatrix([[U,c_s],[c_s,U]])
x = np.linspace(0,1,N)

def p0(x,sigma=sigma):
    return np.exp(-x**2/sigma**2)
                 
def p1 (x,p0=p0,sigma=sigma,k=k):
    return p0(x)*np.cos(k*x/sigma)

def p(x,p0=p0,p1=p1,x0=x0,x1=x1,coeff=0.1):
    return p0(x-x0) + coeff*p1(x-x1)

p_n = p(x)
u_n = np.zeros(N)
p_n1 = np.zeros(N)
u_n1 = np.zeros(N)

def deriv(f,i,h,x):
    if callable(f):
        return (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h) )/(12*h)
    else:
        if i == 0:
            return (-f[i+2] + 8*f[i+1] - 8*f[-1] + f[-2] )/(12*h)
        elif i == 1:
            return (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[-1] )/(12*h)
        elif i == len(x)-1:
            return (-f[1] + 8*f[0] - 8*f[i-1] + f[i-2] )/(12*h)
        elif i == len(x)-2:
            return (-f[0] + 8*f[i+1] - 8*f[i-1] + f[i-2] )/(12*h)
        else:
            return (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2] )/(12*h)
    
def dd(f,x,h,deriv=deriv):
    N = len(x)
    df = np.zeros(N)
    if callable(f):
        for i in range(N):
            if i == 0:
                x_temp = np.roll(x,2)
                df[i] = deriv(f,2,h,x_temp[2])
            elif i == 1:
                x_temp = np.roll(x,1)
                df[i] = deriv(f,2,h,x_temp[2])
            elif i == N-2:
                x_temp = np.roll(x,-1)
                df[i] = deriv(f,N-3,h,x_temp[N-3])
            elif i == N-1:
                x_temp = np.roll(x,-2)
                df[i] = deriv(f,N-3,h,x_temp[N-3])
            else:
                df[i] = deriv(f,i,h,x[i])
        return df
    else:
        for i in range(N):
            """
            if i == 0:
                f_temp = np.roll(f,2)
                df[i] = deriv(f_temp,2,h,True)
            elif i == 1:
                f_temp = np.roll(f,1)
                df[i] = deriv(f_temp,2,h,True)
            elif i == N-2:
                f_temp = np.roll(f,-1)
                df[i] = deriv(f_temp,N-3,h,True)
            elif i == N-1:
                f_temp = np.roll(f,-2)
                df[i] = deriv(f_temp,N-3,h,True)
            else:
            """
            df[i] = deriv(f,i,h,x)
        return df     

#x = np.arange(7)
def f(x):
    return np.cos(2*np.pi*x)


dt = 0.001
T = 0
T1 =1
NT = int(T1/dt)
sol = np.zeros([2,NT+2,N+1])
p_x = dd(p,x,x[1]-x[0])
u_x = dd(u_n,x,x[1]-x[0])
sol[0,0,:]= np.append(u_n,u_n[0])
sol[1,0,:]= np.append(p_n,p_n[0])
i = 1

pu = np.asarray([u_n,p_n])
pu_1=np.asarray([u_n1,p_n1])
def rhs(y,x=x,U=U,c_s=c_s,dd=dd):
    u_x = dd(y[0],x,x[1]-x[0])
    p_x = dd(y[1],x,x[1]-x[0])
    return np.asarray([U*u_x+c_s*p_x,U*p_x+c_s*u_x])


def func(x,pu,dx,dt,rhs):
    print('shape of x is: ' + str(np.shape(pu)))
    return -x+pu-dt*rhs(x)

from scipy.optimize import fsolve #only works with flattened arrays
v_sol = np.zeros(np.shape(pu))

while T <=T1:
    #u_n1 = u_n - dt*(U*u_x+c_s*p_x)
    #p_n1 = p_n - dt*(U*p_x+c_s*u_x)
    pu1 = pu - dt*rhs(pu)
    #pu1 = fsolve(func,v_sol,args=(pu,x[1]-x[0],dt,rhs))
    u_n1 = pu1[0]
    p_n1 = pu1[1]
    sol[0,i,:]=np.append(u_n1,u_n1[0])
    sol[1,i,:]=np.append(p_n1,p_n1[0])
    #u_n1[-1] = u_n1[0]
    #p_n1[-1] = p_n1[0]
    #u_n = u_n1
    #p_n = p_n1
    pu[0] = u_n1
    pu[1] = p_n1
    #u_x = dd(u_n,x,x[1]-x[0])
    #p_x = dd(p_n,x,x[1]-x[0])
    i += 1
    T += dt
    

fig, ax = plt.subplots(figsize=(10, 6))
ax.set(xlim=(-0.01, 1.01), ylim=(-2, 2))

line = ax.plot(np.append(x,x[-1]+x[-1]-x[-2]), sol[0,0,:], color='k', lw=1)[0]
line2 = ax.plot(np.append(x,x[-1]+x[-1]-x[-2]),sol[1,0,:],color='C1',lw=1)[0]

def animate(i):
    line.set_ydata(sol[0,i,:])
    line2.set_ydata(sol[1,i,:])
    
anim = FuncAnimation(
    fig, animate, interval=5, frames=NT)
 
plt.draw()
plt.show()