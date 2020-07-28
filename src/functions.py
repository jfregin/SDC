#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:19:24 2020

@author: jf
TODO:
    -extract DC-implicit and SDC-implicit methods and add them to the fcs class
    -remove unnecessary functions (DC2 or DC) best would be to unify DC-EE and DC-IE into one function
"""
class fcs:
    import numpy as np
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
        """
        calculates lagrange polynomial
        """
        import numpy as np
        n = len(xm) - 1
        lagrpoly = np.array([lagrange(x, i, xm) for i in range(n + 1)])
        y = np.dot(ym, lagrpoly)
        return y

    def jacobian(f,t,y,dy=1e-6):
        """
        Calculates Jacobian Matrix of function f(t,y) at points (t,y)
        
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
    
    def newton(f,t,y0,jacobian=jacobian,acc = 1e-10):
        import numpy as np
        import copy
        i = 0
        x = copy.deepcopy(y0)
        while np.sqrt(np.dot(np.asarray(f(t, x)), np.asarray(f(t, x)))) > acc:
            if i > 1000:
                print('max iterations exceeded')
                break
            Df = jacobian(f, 0, x)
            Df_i = np.linalg.inv(Df)
            feval = -np.asarray(f(0, x))
            try:
                s0 = Df_i @ feval
            except:
                s0 = Df_i * feval
            
            #print(x)
            #print(Df_i)
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
        iffunction: specify wether a function or an array is passed as
                    the RHS of the IVP
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
        dx1 = f_x(t, y+delta) # y+ delta for all functions this needs to be changed!
        for i in range(0, iterations):
            print('i:' + str(i))
            dx2 = polyderiv(t, t, y)
            print(dx2)
            deltad = dx1 - dx2
            delta, t = integrator(deltad, a, b, 0, dt, False)
            y = y + delta
        return y, t, delta, dx1, dx2

    def deferred_correction2(f_x,a,b,y0,dt,iterations=3,integrator=euler_explicit,polyderiv=polyintd,equispaced=True):
        """
        numerical integration of f_x in the interval [a,b] starting at [a,y0]
        using deferred corrections.
        Does NOT work yet with functions that depend on F = int f_x iteself
        because there is no way to include delta into integrator yet
        """
        import numpy as np
        dx2 = 0
        dx1 = 0
        y, t = integrator(f_x, a, b, y0, dt,equispaced=equispaced)
        for i in range(0, iterations):
            def deltafunction(time,x,f=f_x,delta=0,polyderiv=polyderiv,tt=t,y=y):
                dx1 = f(time,x+delta)
                dx2 = polyderiv(time,tt,y)
                print(dx2)
                deltad = dx1-dx2
                return deltad
            tt, yy = a, 0
            sol = yy
            ttt = tt
            if equispaced==True:
                while tt < b:
                    try:
                        yy = yy + dt * deltafunction(tt, yy,delta=yy)
                        sol = np.append(sol, yy)
                        tt += dt
                        ttt = np.append(ttt, tt)
                    except Exception as ex:
                        print(ex)
            else:
                j=1
                while tt < b:
                    try:
                        ddt= dt[j]-dt[j-1] 
                        yy = yy + ddt * deltafunction(tt, yy,delta=yy)
                        sol = np.append(sol, yy)
                        tt += ddt
                        ttt = np.append(ttt, tt)
                        j+=1
                    except Exception as ex:
                        print(ex)
            y = y + sol
        return y, t, 1, dx1, dx2
            
    
    def coord_trans(cin,a,b,A=-1,B=1):
        """
        transforms coordinates cin within the interval [A,B] 
        to coordinates cout within a new interval [a,b]

        Parameters
        ----------
        cin : coordinates within interval [A,B]
        a : lower boundary of new interval
        b : upper boundary of new interval
        A : initial interval lower boundary, default is A=-1
        B : initial interval upper boundary, default is B=1

        Returns
        -------
        cout : coordinates within new interval [a,b]
        """
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
        #print('weights: ' +str(weights))
        return eval(f,nodes,weights,pw=pw)
    


    def Newton_polynomial_specific(x,t='',y='',NVM= NewtonVM,HN = Horner_newton):
        """
        input: see NewtonVM
        returns two arrays
        x: 
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
        GL function calculating Gauss Legendre nodes and weights
        NP function evaluating Lagrange polynomial specific time 
        GQ gaussian quadrature
        TODO:
            equispace must be false -> remove or change
            adapt for different integrators e.g. delta = integrator(*args)
            
        """
        import numpy as np
        N = len(dt) - 2 #number of nodes between a and b
        y, t = integrator(f, a, b, y0, dt, equispaced=equispaced) #initial trajectory
        #print('y_f: '+str(y))
        #y[:]=y0
        for j in range(0, iterations):
            epsilon = np.zeros(len(t))
            epsilon[0] = y0
            for i in range(1, len(t)):
                nodes, notneeded = GL(N, a, t[i])
                #print('t '+str(t))
                #print('nodes '+str(nodes))
                notneeded, specific = NP(nodes, t, f(t, y)) #specific = newton basis polynomial at nodes
                #print('NP: '+str(specific))
                #print('gb: '+str(garbage))
                #print('f(t,y): '+str(f(t,y)))
                # integrate polynomial from a to t[i], M (number of Gauss Legendre nodes) could be chosen arbitrarily 
                epsilon[i] = GQ(specific, a, (t[i]), M=N) + y0 # this could be improved, because GL was already called and will be again within GQ
            
            #calculate epsilon
            #print(epsilon)
            epsilon = epsilon - y
            #print('eps: '+str(epsilon))
            # now everything is known to calculate delta
            delta = np.zeros(len(t))
            for i in range(1, len(t)):
                dt = t[i] - t[(i - 1)]
                delta[i] = delta[(i - 1)] + dt * (f(t[(i - 1)], y[(i - 1)] + delta[(i - 1)]) - f(t[(i - 1)], y[(i - 1)])) + epsilon[i] - epsilon[(i - 1)]
                #print('e1-e0: '+str(epsilon[i] - epsilon[(i - 1)]))
            y = y + delta
        return t, y,epsilon
    
    
    
    #implicit euler
    def implicit_euler(f,a,b,y0,dt,newton=newton,equispaced=True):
        """
        implicit/backward Euler method

        Parameters
        ----------
        f : function to integrate
        a : lower boundary (integer)
        b : upper boundary (integer)
        y0 :initial value(s) (np.array or list)
        dt : timestep
        newton : method to calculate at new time step
                default is newton method, but any other will do

        Returns
        -------
        solution (np.array), timeslices (np.array)

        """
        import numpy as np
        y = np.asarray(y0)
        t = a
        tt = t    
        sol = list([y])
        i = 0
        if equispaced:
            def ff1(t,x,f=f,y=y,dt = dt):
                x = np.asarray(x)
                y = np.asarray(y)
                return x-y-dt*np.asarray(f(t,x))
            while t < b-dt/4:
                t = t+dt
                tt = np.append(tt,t)
                y_s1 = newton(ff1, t,sol[i])
                y = sol[i] + dt*np.asarray(f(t,y_s1)) # not needed just for checking...
                sol = np.append(sol,list([y]),axis=0)
                i += 1
                ff1.__defaults__ = (f,y,dt) #update y in defaults|is necessary for newton to work properly
            return sol[:,0],tt
        else:
            dt_default = dt[1]-dt[0]
            def ff1(t,x,f=f,y=y,dt = dt_default):
                x = np.asarray(x)
                y = np.asarray(y)
                return x-y-dt*np.asarray(f(t,x))
            for i in range(0,len(dt)-1):
                ddt = dt[i+1] -dt[i]
                y_s1 = newton(ff1, dt[i+1],sol[i])
                y = sol[i] + ddt*np.asarray(f(t,y_s1)) # not needed just for checking...
                sol = np.append(sol,list([y]),axis=0)
                i += 1
                ff1.__defaults__ = (f,y,ddt) #update y in defaults|is necessary for newton to work properly
            return sol[:,0], dt
        
    def nodes_weights(n,a,b,A=-1,B=1):
        # nodes and weights for gauss legendre quadrature
        from scipy.special import legendre
        import numpy as np
        poly = legendre(n)
        polyd = poly.deriv()
        nodes= poly.roots
        nodes = np.sort(nodes)
        weights = 2/((1-nodes**2)*(np.polyval(polyd,nodes))**2)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights=(b-a)/(B-A)*weights
        return nodes, weights

    def my_get_weights(n,a,b,nodes,nodes_weights=nodes_weights,NewtonVM=NewtonVM,Horner_newton=Horner_newton,eval=eval):
        # integrates lagrange polynomials to the points [nodes]
        import scipy
        import numpy as np
        nodes_m, weights_m= nodes_weights(np.ceil(n/2),a,b) # use gauss-legendre quadrature to integrate polynomials
        weights = np.zeros(n)
        for j in np.arange(n):
            coeff = np.zeros(n)
            coeff[j] = 1.0 # is unity because it needs to be scaled with y_j for interpolation we have  sum y_j*l_j
            poly_coeffs = scipy.linalg.solve_triangular(NewtonVM(nodes),coeff,lower=True)
            eval_newt_poly = Horner_newton(poly_coeffs, nodes, nodes_m)
            weights[j] = eval(eval_newt_poly,nodes_m,weights_m) #nodes_m are not needed...
        return weights

    def lnw(n,a,b,A=-1,B=1):
        import numpy as np
        from scipy.special import legendre
        # nodes and weights for gauss - lobatto quadrature # See Abramowitz & Stegun p 888
        nodes=np.zeros(n)
        nodes[0]=A
        nodes[-1]=B
        poly = legendre(n-1)
        polyd = poly.deriv()
        subnodes = np.sort(polyd.roots)
        nodes[1:-1]=subnodes
        weights = 2/(n*(n-1)*(np.polyval(poly,nodes))**2)
        weights[0] =2/(n*(n-1)) 
        weights[-1] =2/(n*(n-1)) 
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        #nodes = a + (b-a)/2*(1+nodes)
        weights=(b-a)/(B-A)*weights
        return nodes, weights
    
    #radau
    def rnw(n,a,b,A=-1,B=1):
        import numpy as np
        from scipy.special import legendre
        # nodes and weights for gauss - radau quadrature # See Abramowitz & Stegun p 888
        nodes=np.zeros(n)
        nodes[0]=A
        #nodes[-1]=B
        p = np.poly1d([1,1])
        pn = legendre(n)
        pn1= legendre(n-1)
        poly,remainder = (pn + pn1)/p # [1] returns remainder from polynomial division
        #print('Polynomial division remainder: '+str(remainder(0))) #CAREFUL: remainder increases with n
        nodes[1:] = np.sort(poly.roots)
        weights = 1/n**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/n**2,weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights=(b-a)/(B-A)*weights
        return nodes, weights
    
    def rnw_r(n,a,b,A=-1,B=1):
        import numpy as np
        from scipy.special import legendre
        #        DOES NOT WORK YET
        nodes[0]=A
        #nodes[-1]=B
        p = np.poly1d([1,1])
        pn = legendre(n)
        pn1= legendre(n-1)
        poly,remainder = (pn + pn1)/p # [1] returns remainder from polynomial division
        print('Polynomial division remainder: '+str(remainder(0))) #CAREFUL: remainder increases with n
        nodes[1:] = np.sort(poly.roots)
        nodes = (1-nodes)[::-1]
        weights = 1/n**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/n**2,weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights=(b-a)/(B-A)*weights
        return nodes, weights
    
    def Qmatrix(nodes,a,my_get_weights=my_get_weights):
        """
        Integration Matrix 
        """
        import numpy as np
        M = len(nodes)
        Q = np.zeros([M, M])
    
        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            w = my_get_weights(M, a, nodes[m],nodes)
            Q[m, 0:] = w
    
        return Q
    
    def Smatrix(Q):
        """
        Integration matrix based on Q: sum(S@vector) returns integration
        """
        from copy import deepcopy
        import numpy as np
        M = len(Q)
        S = np.zeros([M, M])
    
        S[0, :] = deepcopy(Q[0, :])
        for m in np.arange(1, M):
            S[m, :] = Q[m, :] - Q[m - 1, :]
    
        return S


    def SDC_S(f,a,b,y0,dt,iterations=3,equispaced=False,integrator=euler_explicit,GL=GL,GQ=GQ,NP=Newton_polynomial_specific,getQ = Qmatrix,getS = Smatrix,gw = my_get_weights):   
        """
        Calculates trajectory of dynamical systems using SDC method
        gw individual weights for Qmatrix (integrates polynomials)
        NP function evaluating Lagrange polynomial specific time 
        GQ gaussian quadrature
        Note that dt must contain ALL timesteps if equispaced = False
        TODO:
            adapt for different integrators e.g. delta = integrator(*args)
            
        """
        import numpy as np
        y, t = integrator(f, a, b, y0, dt, equispaced=equispaced) #initial trajectory
        Q = getQ(t[1:-1],a,gw)
        S = getS(Q)
        y = y[0:-1] # don't need f(t_n) now
        for j in range(0, iterations):
            rhs = f(t[1:-1], y[1:])
            #calculate residual (sigma in Dutt)
            resi = np.cumsum(S@rhs)
            resi  = np.append(0, resi)
            resi = resi + y0 - y
            # now everything is known to calculate delta
            delta = np.zeros(len(t)-1)
            for i in range(1, len(t)-1):
                dt = t[i] - t[(i - 1)]
                delta[i] = delta[(i - 1)] + dt * (f(t[(i - 1)], y[(i - 1)] + delta[(i - 1)]) - f(t[(i - 1)], y[(i - 1)])) + resi[i] - resi[(i - 1)]
            y = y + delta
        y = np.append(y, NP(t[-1], t[0:-1], y)[1])
        return t, y, resi
    
    def SDC_Direct(f,a,b,y0,dt,iterations=3,equispaced=False,integrator=euler_explicit,GL=GL,GQ=GQ,NP=Newton_polynomial_specific,getQ = Qmatrix,getS = Smatrix,gw = my_get_weights,test=True):   
        """
        Calculates trajectory of dynamical systems using SDC method
        gw individual weights for Qmatrix (integrates polynomials)
        NP function evaluating Lagrange polynomial specific time 
        GQ gaussian quadrature
        Note that dt must contain ALL timesteps if equispaced = False
        TODO:
            adapt for different integrators e.g. delta = integrator(*args)
            
        """
        import numpy as np
        from copy import deepcopy
        y, t = integrator(f, a, b, y0, dt, equispaced=equispaced) #initial trajectory
        Q = getQ(t[1:-1],a,gw)
        #Id = np.eye(len(Q))
        #lambd=-1
        #A=(Id - lambd*Q)
        #B = np.zeros(len(Q))
        #B[:] = np.exp(0)
        #Sol = np.linalg.solve(A,B) # works and returns correct solution
        S = getS(Q)
        y = y[0:-1] # don't need f(t_n) now
        #y[1:] = Sol
        y_new = deepcopy(y)
        y_old = deepcopy(y)
        #print(y)
        for j in range(0, iterations):
            rhs = f(t[1:-1], y[1:])
            #calculate residual (sigma in Dutt)
            #print(np.shape(S))
            resi = np.cumsum(S@rhs)
            resi2 = S@rhs
            resi  = np.append(0, resi)
            resi = resi + y0 - y
            res_out = Q@f(t[1:-1],y_new[1:])
            res_out = np.append(0,res_out)
            res_out = res_out +y0 -y_new
            #print('S: '+str(res_out))
            #print('D: '+str(resi))
            #print('resi: '+str(resi))
            # now everything is known to calculate delta
            delta = np.zeros(len(t)-1)
            for i in range(1, len(t)-1):
            #    resi2 = 0
            #    for k in range(0,len(S)):
            #        resi2 = resi2 + S[i-1,k]*f(t[i],Sol[i-1])
                dt = t[i] - t[(i - 1)]
                #print('resi2:'+str(resi2))
                #print('match:'+ str(dt*(f(t[i-1],y_new[i-1]) -f(t[i-1],y_old[i-1]))))
                y_new[i] = y_new[i-1] + dt*(f(t[i-1],y_new[i-1]) -f(t[i-1],y_old[i-1])) + resi2[i-1]
                #y_new[i] = y_new[i] + dt*(f(t[i-1],y_new[i-1]) -f(t[i-1],y_old[i-1])) + (resi[i] - resi[(i - 1)])
                delta[i] = delta[(i - 1)] + dt * (f(t[(i - 1)], y[(i - 1)] + delta[(i - 1)]) - f(t[(i - 1)], y[(i - 1)])) + resi[i] - resi[(i - 1)]
                #if i ==1:
                    #print('S: '+str(y_new[i]))
                    #print('D: '+str(delta[i]))
            y = y + delta
            y_old = deepcopy(y_new)
        #y = np.append(y, NP(t[-1], t[0:-1], y)[1])
        y_new = np.append(y_new, NP(t[-1], t[0:-1], y_new)[1])
        return t, y_new, res_out,y

    def SDC_D2(f,a,b,y0,dt,iterations=3,equispaced=False,integrator=euler_explicit,GL=GL,GQ=GQ,NP=Newton_polynomial_specific,getQ = Qmatrix,getS = Smatrix,gw = my_get_weights,test=False):   
        """
        Calculates trajectory of dynamical systems using SDC method
        gw individual weights for Qmatrix (integrates polynomials)
        NP function evaluating Lagrange polynomial specific time 
        GQ gaussian quadrature
        Note that dt must contain ALL timesteps if equispaced = False
        TODO:
            adapt for different integrators e.g. delta = integrator(*args)
            
        """
        import numpy as np
        from copy import deepcopy
        y_new, t = integrator(f, a, b, y0, dt, equispaced=equispaced) #initial trajectory
        Q = getQ(t[1:-1],a,gw)
        #Id = np.eye(len(Q))
        #lambd=-1
        #A=(Id - lambd*Q)
        #B = np.zeros(len(Q))
        #B[:] = np.exp(0)
        #Sol = np.linalg.solve(A,B) # works and returns correct solution
        S = getS(Q)
        y_new = y_new[0:-1] # don't need f(t_n) now
        y_old = deepcopy(y_new)
        for j in range(0, iterations):
            rhs = f(t[1:-1], y_new[1:])
            resi2 = S@rhs
            res_out = Q@f(t[1:-1],y_new[1:])
            res_out = np.append(0,res_out)
            res_out = res_out +y0 -y_new
            for i in range(1, len(t)-1):
                dt = t[i] - t[(i - 1)]
                y_new[i] = y_new[i-1] + dt*(f(t[i-1],y_new[i-1]) -f(t[i-1],y_old[i-1])) + resi2[i-1]
            y_old = deepcopy(y_new)
        y_new = np.append(y_new, NP(t[-1], t[0:-1], y_new)[1])
        return t, y_new, res_out
    
    def SDC_D2_implicit(f,a,b,y0,dt,iterations=3,equispaced=False,integrator=implicit_euler,GL=GL,GQ=GQ,NP=Newton_polynomial_specific,getQ = Qmatrix,getS = Smatrix,gw = my_get_weights,newton=newton,test=True):   
        """
        Calculates trajectory of dynamical systems using SDC method
        gw individual weights for Qmatrix (integrates polynomials)
        NP function evaluating Lagrange polynomial specific time 
        GQ gaussian quadrature
        Note that dt must contain ALL timesteps if equispaced = False
        TODO:
            adapt for different integrators e.g. delta = integrator(*args)
            
        """
        import numpy as np
        from copy import deepcopy
        y_new, t = integrator(f, a, b, y0, dt, equispaced=equispaced) #initial trajectory
        Q = getQ(t[1:-1],a,gw)
        S = getS(Q)
        
        #for testing purposes only solution shall not change if sol is used as y
        #Id = np.eye(len(Q))
        #lambd=-1
        #A=(Id - lambd*Q)
        #B = np.zeros(len(Q))
        #B[:] = np.exp(0)
        #Sol = np.linalg.solve(A,B) # works and returns correct solution
        
        # initialize y_new (new iteration) and y_old (previous iteration)
        y_new = y_new[0:-1] # don't need f(t_n) now
        y_old = deepcopy(y_new)
        
        #calculate residual | res_out is for output ONLY and could be removed.
        rhs = f(t[1:-1], y_new[1:])
        resi2 = S@rhs
        res_out = Q@f(t[1:-1],y_new[1:])
        res_out = np.append(0,res_out)
        res_out = res_out + y0 - y_new
        
        dt_default = t[1]-t[0]
        def ff1(t,x,f=f,y=y_new[0],y_old=y_old[1],dt = dt_default,res=resi2[0]):
            x = np.asarray(x)
            y = np.asarray(y)
            return x - y - dt*(f(t,x) - f(t,y_old))- res
        for j in range(0, iterations):
            ff1.__defaults__ = (f,y_new[0],y_old[1],dt_default,resi2[0])
            for i in range(1, len(t)-2):
                #dt = t[i] - t[(i - 1)]
                #y_new[i] = y_new[i-1] + dt*(f(t[i-1],y_new[i-1]) -f(t[i-1],y_old[i-1])) + resi2[i-1] #resi is [i-1] because there is no "0" entry starts at resi[0] corresponds to t[1]
                #y_new[i] = y_new[i-1] + dt*(f(t[i],y_new[i]) - f(t[i],y_old[i]))+ resi2[i-1]
                y_new[i] = newton(ff1, t[i],y_new[i-1])
                ff1.__defaults__ = (f,y_new[i],y_old[i+1],t[i+1]-t[i],resi2[i]) #update defaults 
                print(y_new[i])
            y_new[i+1] = newton(ff1, t[i+1],y_new[i])  # calculate y at t[-1]
            rhs = f(t[1:-1], y_new[1:])
            resi2 = S@rhs
            res_out = Q@f(t[1:-1],y_new[1:])
            res_out = np.append(0,res_out)
            res_out = res_out +y0 -y_new
            y_old = deepcopy(y_new)

        y_new = np.append(y_new, NP(t[-1], t[0:-1], y_new)[1])
        return t, y_new, res_out
 
    def IMEX(ff,fs,y0,t,newton=newton):
        import numpy as np
        n = len(t)
        y = np.zeros(n)
        y[0] = y0
        def ff1(t,x,f=ff,y=y0,dt = t[1]-t[0]):
            x = np.asarray(x)
            y = np.asarray(y)
            return x-y-dt*np.asarray(f(t,x))
        
        for i in range(1,n):
            dt = t[i] -t[i-1]
            y[i] = y[i-1] + dt*fs(t[i-1],y[i-1]) # explicit part
            ff1.__defaults__ = (ff,y[i],dt)
            y[i] = newton(ff1, t[i],y[i-1]) #implicit part
        return y
    
    def SDC_FWSW(f,ff,fs,a,b,y0,dt,iterations=3,equispaced=False,integrator=IMEX,GL=GL,GQ=GQ,NP=Newton_polynomial_specific,getQ = Qmatrix,getS = Smatrix,gw = my_get_weights,newton=newton,test=True):   
        """
        Calculates trajectory of dynamical systems using SDC method
        gw individual weights for Qmatrix (integrates polynomials)
        NP function evaluating Lagrange polynomial specific time 
        GQ gaussian quadrature
        Note that dt must contain ALL timesteps if equispaced = False
        TODO:
            adapt for different integrators e.g. delta = integrator(*args)
            
        """
        import numpy as np
        from copy import deepcopy
        import matplotlib.pyplot as plt
        y_new = integrator(ff,fs,y0, dt) #initial trajectory
        t = dt
        Q = getQ(t[1:-1],a,gw)
        S = getS(Q)
        #for testing purposes only solution shall not change if sol is used as y
        #Id = np.eye(len(Q))
        #lambd=-1
        #A=(Id - lambd*Q)
        #B = np.zeros(len(Q))
        #B[:] = np.exp(0)
        #Sol = np.linalg.solve(A,B) # works and returns correct solution
        
        # initialize y_new (new iteration) and y_old (previous iteration)
        y_new = y_new[0:-1] # don't need f(t_n) now
        y_old = deepcopy(y_new)
        
        #calculate residual | res_out is for output ONLY and could be removed.
        rhs = f(t[1:-1], y_new[1:])
        resi2 = S@rhs
        res_out = Q@f(t[1:-1],y_new[1:])
        res_out = np.append(0,res_out)
        res_out = res_out + y0 - y_new
        
        dt_default = t[1]-t[0]
        def ff1(t,x,f=ff,y=y_new[0],y_old=y_old[1],dt = dt_default):
            x = np.asarray(x)
            y = np.asarray(y)
            return x - y - dt*(f(t,x) - f(t,y_old))
        for j in range(0, iterations):
            
            for i in range(1, len(t)-2):
                deltat = t[i]-t[i-1]
                y_new[i] = y_new[i-1] + deltat*(fs(t[i-1],y_new[i-1]) -fs(t[i-1],y_old[i-1])) + resi2[i-1]
                ff1.__defaults__ = (ff,y_new[i],y_old[i],deltat)
                y_new[i] = newton(ff1, t[i],y_new[i-1])
                #print(y_new[i])
                
            deltat = t[i+1]-t[i]
            y_new[i+1] = y_new[i] + deltat*(fs(t[i],y_new[i]) -fs(t[i],y_old[i])) + resi2[i]
            ff1.__defaults__ = (ff,y_new[i+1],y_old[i+1],deltat) #update defaults 
            y_new[i+1] = newton(ff1, t[i+1],y_new[i])  # calculate y at t[-1]
            rhs = f(t[1:-1], y_new[1:])
            resi2 = S@rhs
            res_out = Q@f(t[1:-1],y_new[1:])
            res_out = np.append(0,res_out)
            res_out = res_out +y0 -y_new
            y_old = deepcopy(y_new)
        y_new = np.append(y_new, NP(t[-1], t[0:-1], y_new)[1])
        return t, y_new, res_out
    
    def Q_fast(t):
        """
        EQ. 3.6 SISDC Paper
        """
        import numpy as np
        dim = len(t)-2
        diff = np.diff(t[0:-1])
        Q_f = np.zeros([dim,dim])
        for i in range(0,dim):
            Q_f[i,0:i+1] = diff[0:i+1]
        delta_t = t[-1]-t[0]
        Q_f = 1/delta_t * Q_f
        return Q_f
    
    def Q_slow(t):
        """
        EQ. 3.7 SISDC Paper
        """
        import numpy as np
        dim = len(t)-2
        diff = np.diff(t[0:-1])
        Q_s = np.zeros([dim,dim]) 
        for i in range(1,dim):
            Q_s[i,0:i] = diff[0:i]
        delta_t = t[-1]-t[0]
        Q_s = 1/delta_t * Q_s
        return Q_s
    
    def Q_slow_alt(t,Qf = Q_fast):
        import numpy as np
        Q_f = Qf(t)
        Q_s = np.zeros(Q_f.shape)
        Q_s[1:,:-1] = Q_f[1:,1:]
        return Q_s
    
    def IMEX_matrix(lf,ls,t):
        import numpy as np
        dim = len(t)-2
        diff = np.diff(t[0:-1])
        R = np.zeros([dim,dim],dtype=np.complex)
        terms = (1+1j*diff*ls)/(1-1j*diff*lf)
        diag = np.cumprod(terms)
        np.fill_diagonal(R,diag)
        return R
    
    def stabl(lf,ls,t,k=3,gw=my_get_weights,IMEX_matrix=IMEX_matrix):
        from numpy.linalg import matrix_power as mp
        import numpy as np
        #u0 = u_init[1:-1]
        dim = len(t)-2
        a = t[0]
        b = t[-1]
        delta_t = b-a
        nodes = t[1:-1]
        Q_f  = fcs.Q_fast(t)
        Q_s  = fcs.Q_slow_alt(t)
        Q = fcs.Qmatrix(nodes,a,gw)
        Q_delta = 1/(b-a)*Q
        Id = np.identity(dim,dtype=np.complex)
        L = Id - delta_t*(1j*lf*Q_f + 1j*ls*Q_s)
        L_inv = np.linalg.inv(L)
        R = delta_t*((1j*(lf+ls)*Q_delta)-(1j*lf*Q_f + 1j*ls*Q_s))
        garb, q = fcs.rnw(dim,a,b)
        q= q[::-1]
        R2 = IMEX_matrix(lf,ls,t)
        term = mp(L_inv@R,k)
        term2 = mp(L_inv@R,k)@R2
        for j in range(0,k):
            term +=  mp(L_inv@R,j)@L_inv
            term2 += mp(L_inv@R,j)@L_inv
        #term = mp(L_inv@R,k) + sum_term
    
        ones = np.ones(len(Q))
        return 1 + 1j*(lf+ls)*q.dot(term.dot(ones)), 1 + 1j*(lf+ls)*q.dot(term2.dot(ones))