#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:19:24 2020

@author: jf
"""
class fcs:
    
    def lagrange_derivative(x,i,xm):
        """
        Evaluates the derivative of the i-th Lagrange polynomial at x
        based on grid data xm
        """
        n = len(xm) - 1
        l = 0
        for k in range(n + 1):
            if k != i:
                inc = 1 / (xm[i] - xm[k])
                for j in range(n + 1):
                    if i != j and k != j:
                        inc = inc * (x - xm[j]) / (xm[i] - xm[j])
                l = l + inc
        return l
    
    def lagrange(x,i,xm):
        """
        Evaluates the i-th Lagrange polynomial at x
        based on grid data xm
        """
        n = len(xm) - 1
        y = 1
        for j in range(n + 1):
            if i != j:
                y *= (x - xm[j]) / (xm[i] - xm[j])   
        return y
    
    def subinterval(a,b,num_int):
        """
        Parameters
        ----------
        a : real
            lower boundary
        b : real
            upper boundary
        num_int : integer
            number of grid points between a and b
        
        Returns
        -------
        sub_int : new grid with num_int subintervals in between two points
        in the grid. Type:(array)
        """
        import numpy as np
        n = num_int + 1
        step = (b - a) / n
        sub_int = np.arange(a, b, step)
        return sub_int
    
    
    def rk4(sys,a,b,init,step):
        """
        Runge-Kutta fourth order integrator
        
        Parameters
        ----------
        sys : dynamical system
        interval:integer number
        step : time step
        init : initial conditions
        
        Returns
        -------
        y : vector containing the time series of the system
        """
        import numpy as np
        init = np.array(init)
        dim = np.size(init)
        if dim == 1:
            k1, k2, k3, k4 = ([0, 0], [0, 0], [0, 0], [0, 0])
        time = np.arange(a, b + step, step)
        length = len(time)
        y = np.zeros([length, dim])
        y[0, :] = init
        if dim == 1:
            for i in range(length - 1):
                f0 = y[i, :]
                k1[0] = step * np.asarray(sys(time[i], f0[:]))
                k2[0] = step * np.asarray(sys(time[i], f0[:] + k1[0] / 2.0))
                k3[0] = step * np.asarray(sys(time[i], f0[:] + k2[0] / 2.0))
                k4[0] = step * np.asarray(sys(time[i], f0[:] + k3[0]))
                y[i + 1, :] = f0 + 0.16666666666666666 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    
        else:
            for i in range(length - 1):
                f0 = y[i, :]
                k1 = step * np.asarray(sys(time[i], f0[:]))
                k2 = step * np.asarray(sys(time[i], f0[:] + k1[:] / 2.0))
                k3 = step * np.asarray(sys(time[i], f0[:] + k2[:] / 2.0))
                k4 = step * np.asarray(sys(time[i], f0[:] + k3[:]))
                y[i + 1, :] = f0 + 0.16666666666666666 * (k1 + 2 * k2 + 2 * k3 + k4)
    
        return (time, y)
    
    
    def refine_grid(grid1d,subintervals=4,subfunc=subinterval):
        """
        Parameters
        ----------
        grid1d : array
            one dimensional grid array
        subintervals : integer
            number of intervals in between grid points
        subfunc : fuction
            function that creates a grid with subintervals gridpoints between two
            integers a and b. b is not included in the new grid
        Returns
        -------
        new_grid : array
            grid with equally spaced subintervals that includes the starting and
            end points a and b.
        """
        import numpy as np
        si = subintervals
        g = grid1d
        new_grid = subfunc(g[0], g[1], si)
        for i in range(2, len(g)):
            new_grid = np.append(new_grid, subfunc(g[(i - 1)], g[i], si))
        new_grid = np.append(new_grid, g[(-1)])
        return new_grid
    
    def polyintd(x,xm,ym,lagrange_derivative=lagrange_derivative):
        import numpy as np
        n = len(xm) - 1
        lagrpoly = np.array([lagrange_derivative(x, i, xm) for i in range(n + 1)])
        y = np.dot(ym, lagrpoly)
        return y
    
    def polyint(x,xm,ym,lagrange=lagrange):
        import numpy as np
        n = len(xm) - 1
        lagrpoly = np.array([lagrange(x, i, xm) for i in range(n + 1)])
        y = np.dot(ym, lagrpoly)
        return y

    def jacobian(f,t,y,dy=1e-6):
        """
        Parameters
        ----------
        f : TYPE function
            function to calculate jacobian for
        t : TYPE
            time
        y : TYPE list or array containing values at desired point
            [x1,x2,x3,...]
        dy : TYPE, float
            DESCRIPTION. delta to calculate Df. The default is 1e-6.
        
        Returns
        -------
        TYPE
            Jacobian Matrix at point Df(t,y)
        """
        import numpy as np
        y = np.array(y)
        dim = np.size(y)
        jac = np.zeros([dim, dim])
        delta = 2 * dy
        for i in range(0, dim):
            dyv = np.zeros(dim)
            dyv[i] = dy
            F1 = np.asarray(f(t, y + dyv))
            F2 = np.asarray(f(t, y - dyv))
            deriv = (F1 - F2) / delta
            jac[i] = deriv
        
        return jac.T
    
    def newton(f,t,y0,jacobian=jacobian):
        import numpy as np
        import copy
        i = 0
        x = copy.deepcopy(y0)
        while np.sqrt(np.dot(np.asarray(f(t, x)), np.asarray(f(t, x)))) > 1e-16:
            if i > 1000:
                break
            Df = jacobian(f, 0, x)
            Df_i = np.linalg.inv(Df)
            feval = -np.asarray(f(0, x))
            try:
                s0 = Df_i @ feval
            except:
                s0 = Df_i * feval
        
            i += 1
            x += s0
        
        return x   
    
    
            
    def euler_explicit(f,a,b,y0,dt,iffunction=True,equispaced=True):
        """
        Euler forward method:
        f: function 
        a: begin of interval
        b: end of interval
        y0:array containing initial values
        dt:step size
        """
        import numpy as np
        if equispaced:
            t, y = a, y0
            sol = y0
            tt = t
            if iffunction:
                while t < b:
                    try:
                        y = y + dt * f(t, y)
                        sol = np.append(sol, y)
                        t += dt
                        tt = np.append(tt, t)
                    except Exception as ex:
                        print(ex)
                return sol, tt
            else:
                i = 0
                while t < b:
                    try:
                        y += dt * f[i]
                        sol = np.append(sol, y)
                        t += dt
                        tt = np.append(tt, t)
                        i += 1
                    except Exception as ex:
                        print(ex)
                return sol, tt
           
        else:
            t = dt
            y = y0
            sol = y0
            for i in range(1, len(t)):
                dt = t[i] - t[(i - 1)]
                y = y + dt * f(t[(i - 1)], y)
                sol = np.append(sol, y)
        
            return sol, t
        
    def deferred_correction(f_x,a,b,y0,dt,iterations=3,integrator=euler_explicit,polyderiv=polyintd):
        """
        numerical integration of f_x in the interval [a,b] starting at [a,y0]
        using deferred corrections.
        Does NOT work yet with functions that depend on F = int f_x iteself
        because there is no way to include delta into integrator yet
        """
        delta = 0
        dx2 = 0
        dx1 = 0
        y, t = integrator(f_x, a, b, y0, dt)
        dx1 = f_x(t, y)
        for i in range(0, iterations):
            dx2 = polyderiv(t, t, y)
            deltad = dx1 - dx2
            delta, t = integrator(deltad, a, b, 0, dt, False)
            y = y + delta
        
        return y, t, delta, dx1, dx2
    
    def coord_trans(cin,a,b,A=-1,B=1):
        cout = ((b - a) * cin + a * B - b * A) / (B - A)
        return cout

    def NewtonVM(t):
        """
        t: array or list containing nodes.
        returns: array Newton Vandermode Matrix. Entries are in the lower triangle
        Polynomial can be created with
        scipy.linalg.solve_triangular(NewtonVM(t),y,lower=True) where y
        contains the points the polynomial need to pass through
        """
        import numpy as np
        t = np.asarray(t)
        dim = len(t)
        VM = np.zeros([dim, dim])
        VM[:, 0] = 1
        for i in range(1, dim):
            VM[:, i] = (t[:] - t[(i - 1)]) * VM[:, i - 1]
        
        return VM
        
    def Horner_newton(weights,xi,x):
        """
        Horner scheme to evaluate polynomials based on newton basis
        """
        import numpy as np
        y = np.zeros_like(x)
        for i in range(len(weights)):
            y = y * (x - xi[(-i - 1)]) + weights[(-i - 1)]
        
        return y
    
    def GL(M,a,b,ct=coord_trans,A=-1,B=1):
        """
        calculates Gauss-Legendre nodes for Gaussian quadrature at int(M)
        Points between float(a) and float(b).
        A and B shouldn't be changed
        """
        # is a lot faster than GaussLegendre but slower than _getNodes
        import numpy as np
        nodes, weights = np.polynomial.legendre.leggauss(M) # calculate nodes and weights on [-1,1]
        nodes = ct(nodes,a,b) # transform back to interval [a,b]
        weights= (b-a)/(B-A)*weights # also transform weights to [a,b]
        return nodes, weights
    
    def eval(f,nodes,weights,pw=False): # performes gaussian quadrature given the function f
        import types
        import numpy as np
        if pw == False:
            if type(f) == types.FunctionType:
                return np.dot(weights,f(nodes))
            else:
                return np.dot(weights,f)
        else:
            l = len(weights)
            summation = np.zeros(l)
            if type(f) == types.FunctionType:
                for i in range(0,l):
                    summation[i]=np.dot(weights[0:i+1],f(nodes[0:i+1]))
                return summation
            else:
                for i in range(0,l):
                    summation[i] = np.dot(weights[0:i+1],f[0:i+1])
                return summation
            
    
    def GQ(f,a,b,M=10,coord_trans=coord_trans,GL=GL,eval=eval,pw=False):
        """
        performs gaussian quadrature on function f based on the interval [a,b]
        """
        nodes, weights = GL(M,a,b)
        return eval(f,nodes,weights,pw=pw)
    


    def Newton_polynomial_specific(x,t='',y='',NVM= NewtonVM,HN = Horner_newton):
        """
        input: see NewtonVM
        returns two arrays
        x: resolves t in N_linspace steps
        yy: Lagrange Polynomial passing through points(t_i,y_i)
        Note: maybe use *args to prevent t='' and y=''
        """
        from scipy.linalg import solve_triangular
        VM = NVM(t)
        weights = solve_triangular(VM, y, lower=True)
        yy = HN(weights, t, x)
        return x, yy
    
    def Newton_polynomial(t, y, N_linspace=100, NVM= NewtonVM, HN=Horner_newton):
        """
        input: see NewtonVM
        returns two arrays
        x: resolves t in N_linspace steps
        yy: Lagrange Polynomial passing through points(t_i,y_i)
        """
        from scipy.linalg import solve_triangular
        import numpy as np
        VM = NVM(t)
        weights = solve_triangular(VM, y, lower=True)
        x = np.linspace(t[0], t[(-1)], N_linspace)
        yy = HN(weights, t, x)
        return x, yy
    
    
    def SDC(f,a,b,y0,dt,iterations=3,equispaced=False,integrator=euler_explicit,GL=GL,GQ=GQ,NP=Newton_polynomial_specific):   
        """
        Calculates trajectory of dynamical systems using SDC method
        """
        import numpy as np
        N = len(dt) - 2
        y, t = integrator(f, a, b, y0, dt, equispaced=equispaced)
        for j in range(0, iterations):
            epsilon = np.zeros(len(t))
            epsilon[0] = y0
            for i in range(1, len(t)):
                nodes, garbage = GL(N, a, t[i])
                garbage, specific = NP(nodes, t, f(t, y))
                epsilon[i] = GQ(specific, a, (t[i]), M=N) + y0
        
            epsilon = epsilon - y
            delta = np.zeros(len(t))
            for i in range(1, len(t)):
                dt = t[i] - t[(i - 1)]
                delta[i] = delta[(i - 1)] + dt * (f(t[(i - 1)], y[(i - 1)] + delta[(i - 1)]) - f(t[(i - 1)], y[(i - 1)])) + epsilon[i] - epsilon[(i - 1)]
            y = y + delta
        return t, y
    

