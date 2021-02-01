#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:47:48 2021

@author: jf
"""

from functions import fcs as fcs
import numpy as np
from mpmath import findroot

a = 0
b = 0.5

SDC_iterations= 2
SDC_nodes = 3

lf= 3 + 0j
ls = 1 + 0j
cc =1 + 0j
c = np.eye(4,dtype=complex)
m = 1.
c[1,1] = m 
c[3,3] = m*100

k1 = 3
k2= 3
#c = 0.8 + 0j


def F(t,lf=lf,ls = ls, c=cc):
    return np.exp(1j*(lf+ls)*t/c)

def ff(t,y, k1=k1, k2=k2):
    if np.shape(y) == np.shape([0,0,0,0]):
        # if input is just one coordinate
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]
        return np.asarray([x2, -k1*x1 -k2*(x1-x3)+np.sin(0*t), 0, 0])
    else:
        # if input is multiple coordinates
        x1, x2, x3, x4 = y[:,0], y[:,1], y[:,2], y[:,3]
        return np.asarray([x2, -k1*x1 -k2*(x1-x3)+np.sin(0*t), x4*0, x2*0])

def fs(t,y,k1=k1, k2=k2):
    if np.shape(y) == np.shape([0,0,0,0]):
        # if input is just one coordinate
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]
        return np.asarray([0 ,0 , x4, -k2*(x3-x1)])
    else:
        # if input is multiple coordinates
        x1, x2, x3, x4 = y[:,0], y[:,1], y[:,2], y[:,3]
        return np.asarray([x2*0, x1*0, x4, -k2*(x3-x1)])

def f(t,y, ff=ff, fs = fs): 
    return ff(t,y) + fs(t,y)




y0 = np.asarray([1+0j, 1+0j, 1+0j, 1+ 0j])

class O():
    def a(a,b):
        return a + b
    
    def m(a,b):
        return a*b
    
    def norm(a,norm_type):
        from numpy.linalg import norm as norm
        return norm(a, norm_type)
        
    
class U():
    def __add__(self,x):
        return a + b
    
    def m(a,b):
        return a*b
    
    def norm(a,norm_type):
        from numpy.linalg import norm as norm
        return norm(a, norm_type)

# could be just a dictionary; may contain functions if not a linear problem
class PROBLEM():
    def __init__(self, y0, ff, fs, f, c):
        self.y0 = y0
        self.ff = ff
        self.fs = fs
        self.f = f
        self.c = c
        
        
class SOLVER(object):
    import numpy as np
    from functions import fcs as fcs
    
    def __init__(self, object):
        self.p = object             # problem
        self.baseIntegrator = 'IMEX'    # base integrator for SDC
        self.quadrature = 'radau_r'
        self.t = np.nan     #current time
        self.a = np.nan     # time at start
        self.b = np.nan     # time at end
        self.ts= np.nan     # time stamps (points in time where you want to know your solution)
        self.dts = np.nan   # delta_ts
        self.dt = np.nan    # current delta_t
        self.i = 0          # current index
        self.M = np.nan
        self.iterations = np.nan
        self.nodes = np.nan
        self.weights= np.nan
        self.sol = np.nan
        self.Q = np.nan
        self.S = np.nan
        self.tau = np.nan
        self.solution = [self.p.y0,self.p.y0]
        
    # evaluate right hand side + c is shifted to rhs
    def eval_f(self,t,y,f):
        if len(self.p.y0) == 1:
            return O.m(f(t,y),1/self.p.c)
        elif np.shape(y) == np.shape(self.p.y0):
            return np.linalg.inv(c)@f(t,y)
        else:
            ret = np.zeros([len(y),len(self.p.y0)],dtype=complex)
            for i in range(len(y)):
                ret[i] = np.linalg.inv(self.p.c)@f(t[i],y[i])
            return ret.T
                
                
    # setup for sdc  
    def setup(self, ts, SDC_nodes = 8, SDC_iterations = 5, baseIntegrator = 'IMEX'):
        self.ts = ts
        self.a = ts[0]
        self.t = ts[0]
        self.b = ts[-1]
        self.dts = np.diff(self.ts)
        self.dt = self.dts[0]
        self.M = SDC_nodes
        self.iterations = SDC_iterations
        
    # calculate nodes and weights, based on choosen quadrature. TODO: implement further rules    
    def nodes_weights(self):
        if self.quadrature == 'radau_r':           
            self.nodes, self.weights = fcs.rnw_r(self.M ,self.t , self.t + self.dt,A=-1,B=1)    
     
    # IMEX for initial trajectory
    # TODO: implement complex newton (use scipy)
    def IMEX_c(self, t):
        import numpy as np
        n = len(t)
        y = np.zeros(n,dtype=np.complex)
        try:
            if len(self.p.y0) >1:
                y = np.zeros([n,len(self.p.y0)],dtype=complex)
        except:
            y = np.zeros(n,dtype=np.complex)
        y[0] = self.p.y0
        dt_v = np.diff(t)
        self.tau = dt_v
        for i in range(1,n):
            dt = dt_v[i-1]
            #y[i] = np.asarray(findroot(lambda x1, x2: tuple(np.asarray([x1,x2],dtype=complex) - y[i-1] - dt/self.p.c*(self.p.ff(self.t,np.asarray([x1,x2],dtype=complex))+self.p.fs(t,y[i-1]))), (y[i-1][0],y[i-1][1])),dtype=complex)
            y[i] = np.asarray(findroot(lambda x1, x2, x3, x4: tuple(np.asarray([x1,x2, x3, x4],dtype=complex) - y[i-1] - dt*(self.eval_f(self.t,np.asarray([x1,x2, x3, x4],dtype=complex),self.p.ff)+self.eval_f(t,y[i-1],self.p.fs))), (y[i-1][0],y[i-1][1],y[i-1][2],y[i-1][3])),dtype=complex)
        self.sol = y
        
    # one SDC iteration
    def SDC_sweep(self):
        from copy import deepcopy
        Q = self.Q
        S = self.S
        d_tau = self.tau
        
        # append initial 
        y = self.sol
        y_n = deepcopy(y) 
        
        # use own @ routine
        # TODO copy matmul from computer at work and use instead of @
        if np.shape(self.p.y0) == np.shape(1):
            quad = S@self.eval_f(self.nodes,y[1:],self.p.f)
        else:
            l = len(self.p.y0)
            quad = np.zeros([len(S),l], dtype=complex)
            feval = self.eval_f(self.nodes, y[1:],self.p.f)
            for i in range(l):
                quad[:,i] =  S@feval[i]
        for i in range(len(y)-1):
            #function = lambda x1, x2: tuple((y_n[i] + d_tau[i]/self.p.c*(self.p.ff(self.t,np.asarray([x1,x2],dtype=complex))-self.p.ff(self.t,y[i+1]) + self.p.fs(self.t,y_n[i])-self.p.fs(self.t,y[i])) + quad[i]) - np.asarray([x1,x2],dtype=complex))
            function = lambda x1, x2, x3, x4: tuple((y_n[i] + d_tau[i]*(self.eval_f(self.t,np.asarray([x1,x2, x3, x4],dtype=complex),self.p.ff)-self.eval_f(self.t,y[i+1],self.p.ff) + self.eval_f(self.t,y_n[i],self.p.fs)-self.eval_f(self.t,y[i],self.p.fs)) + quad[i]) - np.asarray([x1,x2, x3, x4],dtype=complex))
            y_n[i+1] = np.asarray(findroot(function, (y_n[i][0],y_n[i][1],y_n[i][2],y_n[i][3])),dtype=complex)
        self.sol = y_n
        
        #residual
        self.residual = np.zeros([self.M,len(self.p.y0)],dtype=complex)
        feval = self.eval_f(self.nodes,self.sol[1:],self.p.f)
        #self.residual = self.p.y0 + Q@self.eval_f(self.nodes,self.sol[1:],self.p.f) - self.sol[1:]
        for i in range(0,len(self.p.y0)):
            self.residual[:,i] = self.p.y0[i] + Q@feval[i] - self.sol[1:][:,i]
    # one step forward in time based on timestamps stored in self.ts
    def step(self):
        """
        step forward one timestep

        """
        a = self.t
        b = self.ts[self.i+1]

                
        self.nodes_weights()
        # initial trajectory
        
        self.IMEX_c(np.append(a,self.nodes))
        self.Q = fcs.Qmatrix(self.nodes, a)
        self.S = fcs.Smatrix(self.Q)       
        i = 0
        # SDC below here
        while i < self.iterations:   
            self.SDC_sweep()
            i += 1
        
        # final update
        if np.shape(self.p.y0) == np.shape(1):
            self.y = self.p.y0 + np.dot(self.weights, self.eval_f(self.nodes, self.sol[1:]),self.p.f)
        else:
            l = len(self.p.y0)
            dot = np.zeros(l)
            feval = self.eval_f(self.nodes, self.sol[1:],self.p.f)
            for i in range(l):
                dot[i] = np.dot(self.weights, feval[i])
            self.y = self.p.y0 + dot
        self.i += 1
        self.t = b
        
        #update y0 in problem for next timestep
        self.p.y0 = self.y
        try:
            self.dt = self.dts[self.i]
        except:
            self.dt = self.dts
    
    # step through all timesteps  
    def solve(self):
        while self.t != self.b:
            self.step()
            self.solution.append(self.y)
        self.solution = np.asarray(self.solution[1:])


p = PROBLEM(y0, ff, fs, f, c)
solver = SOLVER(p)
solver.setup(np.linspace(0,100,70),SDC_nodes = SDC_nodes, SDC_iterations = SDC_iterations)
# solve until t_final is reached
solver.solve()
import matplotlib.pyplot as plt
#plt.plot(np.append(0,solver.nodes),F(np.append(0,solver.nodes)), marker='o')
#plt.plot(np.append(0,solver.nodes),solver.sol, marker='x',linestyle='')
#plt.plot([solver.ts[-1],solver.ts[-1]],solver.y,marker='.',linestyle='')
#plt.plot(np.append(0,solver.nodes),np.imag(solver.sol), marker='x',linestyle='')
plt.plot(solver.ts,solver.solution[:,0], color='k', label='x1',linestyle='-')
plt.plot(solver.ts,solver.solution[:,2], color='k', label='x2', linestyle='--')
plt.plot(solver.ts,solver.solution[:,1], color='r', label='u')
plt.plot(solver.ts,solver.solution[:,3], color='r', label='v',linestyle='--')
plt.legend()
#plt.plot(np.append(0,solver.nodes),solver.sol2, marker='x',linestyle='')
print ('max Residual: '+ str(O.norm(solver.residual, np.inf)))

# np.asarray(findroot(lambda x1, x2: tuple(np.asarray([x1,x2],dtype=complex) - y0 - solver.tau[0]/solver.p.c*(solver.p.ff(1,np.asarray([x1,x2],dtype=complex))+solver.p.fs(1,y0))), (y0[0],y0[1])),dtype=complex)