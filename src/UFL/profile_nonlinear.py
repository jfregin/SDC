#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:47:48 2021

@author: jf
"""

from functions import fcs as fcs
import numpy as np
from mpmath import findroot
from firedrake import * 
from SDC_CLASS_UFL import SOLVER
a = 0                           # time of start
b = 3                       # time of end
n = 30                         # number of spatial nodes
n_steps = 10000.
dt = (b-a)/n_steps
M = 3                               # number of GL nodes
#dt = Constant(d_t)
mesh = PeriodicIntervalMesh(n, 1)   # create periodic grid
x = SpatialCoordinate(mesh)[0]      # extract coordinates
c_s = Constant(1)                   # speed of sound
U = Constant(0.05)                  # mean flow

x1 = 0.25
x0 = 0.75
sigma = 0.1
k = 7.2*np.pi

VV = FunctionSpace(mesh, 'GLL', 1)  # function space. Functions will be approximated using first order Lagrange Polynomials
V = VV*VV                                # Create mixed functions space. Need to use due to coupled system


# Setup initial conditions
def p0(x, sigma=sigma):
    return exp(-x**2/sigma**2)
                 
def p1(x, p0=p0, sigma=sigma, k=k):
    return Constant(0)# p0(x)*cos(k*x/sigma)

def p(x, p0=p0, p1=p1, x0=x0, x1=x1, coeff=1.):
    return p0(x-x0) + coeff*p1(x-x1)

p_init = p(x)   

U_n = Function(V)                        # Vector containing Solutions at t
u_n, p_n = U_n.split()
p_n.interpolate(p_init)
#p_n.interpolate(x**0)


SDC_iterations= 2
SDC_nodes = M

# setting up problem functions

def ff1(p_n, c_s):
    return -c_s*p_n.dx(0)

def ff2(u_n, c_s):
    return -c_s*u_n.dx(0)

def fs1(u_n, U):
    return -U*u_n.dx(0)

def fs2(p_n, U):
    return -U*p_n.dx(0)


# could be just a dictionary; may contain functions if not a linear problem
class PROBLEM():
    from firedrake import dx, TrialFunction, TestFunction
    
    def __init__(self, y0, ff1, ff2, fs1, fs2, V, c_s, U):
        self.y0 = y0      # U_n
        self.ff1 = ff1    # basic
        self.ff2 = ff2    # baisc
        self.fs1 = fs1    # basic
        self.fs2 = fs2    # basic
        self.V = V
        self.u_, self.p_ = TrialFunctions(V)
        self.v1, self.v2 = TestFunctions(V)
        self.c_s = c_s
        self.U = U
        self.U1 = Function(V)

    def lhs(self,u_,p_):
        l1 = inner(u_, self.v1)*dx
        l2 = inner(p_, self.v2)*dx
        l = l1 + l2
        return l
        
    def ff(self, u_, p_):
        """
        construct weak form of fast problem (works only for linear problems)
        """
        r1 = inner(self.ff1(p_, self.c_s), self.v1)*dx 
        r2 = inner(self.ff2(u_, self.c_s), self.v2)*dx 
        rhs = r1 + r2
        return rhs

    def fs(self, u_n, p_n):
        """
        construct weak form of slow problem (works only for linear problems)
        """
        r1 = inner(self.fs1(u_n, self.U), self.v1)*dx
        r2 = inner(self.fs2(p_n, self.U), self.v2)*dx
        rhs =  r1 + r2
        return rhs
    
    def f(self, u_n, p_n):
        return self.ff(u_n,p_n) + self.fs(u_n, p_n)


p = PROBLEM(U_n, ff1, ff2, fs1, fs2 , V, c_s, U)
solver = SOLVER(p)
solver.setup(np.linspace(a,b,int(n_steps+1)),SDC_nodes = SDC_nodes, SDC_iterations = SDC_iterations, isnonlinear=False)
# solve until t_final is reached
solver.solve()

