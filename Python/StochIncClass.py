import numpy as np
from scipy.optimize import fsolve, brentq, root, minimize
from quantecon import markov
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from ellipse import estimation
import time
import datetime

class OG(object):
    """An object that solves for the SS and TPI of a given overlapping
    generations model.
    
    Methods
    -------
    get_c0 : Calculate consumption.
    get_c1 : Calculate consumption.
    obj : Calculate euler errors.
    obj2 : Calculate euler errors and return their norm. 
           (This is used for minimization)
    update_Psi_n_S : Update the last rows of the labor policy function.
    update_Psi_b1n0 : Update the labor and savings policy functions.
    update_Psi : Use update_Psi_n_S and update_Psi_b1n0 to update 
                 policy functions.
    update_B_and_N : Update the states B and N.
    set_state : Given B, N determin L, K, r, and w.
    calc_SS : Calculate the steady state.
    calc_TPI : Calculate the time path.
    write_output : Write output file.
    """
    def __init__(self, household_params, firm_params):
        """Instantiate the state parameters of the OG model.

        Attributes
        ----------
        start : some sort of datetime object
        Num : int
              Number of agents in smallest ability group.
        S : int
            Number of time periods in an agent's life.
        J : int
            Number of ability groups.
        beta : float
               Discount factor.
        sigma : float
                Constant of relative risk aversion.
        Pi : (J,J) ndarray
             Probability of movement between ability groups.
        e : (J,) ndarray
            Effective labor units given ability type.
        lambda_bar : (J,) ndarray
                     Steady state ergodic distribution ability of agents.
        M : int
            Total number of agents.
        shocks : (S,M) ndarray
                 Ability life path of all agents.
        theta : float
                Agent elliptical utility labor param.
        ellip_up : float
               Estimated upsilon from elliptical utility package.
        ellip_b : float
               Estimated b from elliptical utility package.                
        A : float
            Firm production multiplier.
        alpha : float
                Cobb-Douglas production parameter.
        delta : float
                Rate of depreciation of capital.
        B : (S,M) array
            Savings decision for M agents, S periods.
        N : (S,M) array
            Labor decisions for M agents, S periods.
        r : float
            Interest rate.
        w : float
            Wage rate.
        grid_size : int
               Number of points in policy function grid..
        b0min : float
               Min endpoint for policy function interpolation.
        b0max : float
               Max endpoint for policy function interpolation.
        Grid : (S,J,grid_size) array
               Policy grids for period s, ability j.
        Psi_b : (S,J,grid_size) array
               Capital policy function for period s, ability j.
        Psi_n : (S,J,grid_size) array
               Labor policy function for period s, ability j.
        euler_errors : list
                       A list of all the errors in the various optimization
                       routines.
        """
        # Start timer for output file.
        self.start = time.time()

        # Make household parameters as attributes.
        (self.Num, self.S, self.J, beta_annual, self.sigma,
         self.Pi, self.e, self.theta) = household_params
        self.ellip_b, self.ellip_up = estimation(self.theta,1.)
        # Convert beta_annual to the value of beta for one period.
        self.beta = beta_annual**(80/self.S)
        
        # Createt the Markov chain object and simulate the shock paths.
        self.MC = markov.core.MarkovChain(self.Pi)
        self.lambda_bar = self.MC.stationary_distributions
        weights = (self.lambda_bar/
            (self.lambda_bar.min())*self.Num).astype('int')
        self.M = np.sum(weights)
        initial_e = np.zeros(self.M)
        for agent in np.cumsum(weights[0][1:]):
            initial_e[agent:] += 1
        self.shocks = self.MC.simulate(self.S,
                                       initial_e.astype('int'),
                                       random_state=1).T
        self.abilities = self.e[self.shocks]
        
        # Set the firm parameters as attributes.
        (self.A, self.alpha, delta_annual) = firm_params
        # Convert delta_annual to the delta for one period.
        self.delta = 1-(1-delta_annual)**(80/self.S)
        
        # Initialize the state.
        self.B = np.empty((self.S, self.M))
        self.B[0] = 0.0
        self.N = np.ones((self.S, self.M))*.9
        # Set initial guesses for r and w.
        self.r, self.w = 0.0514041424518, 1.26652076063 # May need to have good guesses.
        
        # Initialize the policy function objects.
        self.grid_size = 750
        # b0max should be chosen carefully, or else there can be problems.
        self.b0max = 14.0     # Carefully may mean guess and check...
        self.Grid = np.empty((self.S, self.J, self.grid_size))
        for j in range(self.J):
            self.b0min = -self.w*self.e[j]/(1+self.r)+1e-5
            self.Grid[-1,j] = np.linspace(self.b0min, self.b0max, self.grid_size)
        self.Psi_b = np.empty((self.S, self.J, self.grid_size))
        self.Psi_n = np.empty((self.S, self.J, self.grid_size))
        self.Psi_b[-1] = 0.0
        self.euler_errors = []
        self.wSS = ' ' #This is just a fix for some buggy thing I will get to
        
        
    def get_c0(self, b1n0, b0, s, j, i, w, r):
        """Calculate consumption."""
        b1, n0 = b1n0
        return b0*(1+r)+w*self.e[j]*n0-b1
    

    def get_c1(self, b1n0, b0, s, j, i, w, r):
        """Calculate consumption."""
        b1, n0 = b1n0
        b2 = np.array(
             [UnivariateSpline(self.Grid[s+1,j_],
                               self.Psi_b[s+1,j_])(b1) for j_ in xrange(self.J)])
        n1 = np.array(
             [UnivariateSpline(self.Grid[s+1,j_],
                               self.Psi_n[s+1,j_])(b1) for j_ in xrange(self.J)])
        return b1*(1+r)+w*self.e*n1-b2
    

    def obj(self, b1n0, b0, s, j, i, w, r):
        """Calculate the euler errors."""
        b1, n0 = b1n0
        if isinstance(w, (np.ndarray, list)):
            w0, w1 = w
            r0, r1 = r
            c0 = self.get_c0(b1n0, b0, s, j, i, w0, r0)
            c1 = self.get_c1(b1n0, b0, s, j, i, w1, r1)

            err_c = (c0**-self.sigma
                     -self.beta*(1+r1)*np.sum(self.Pi[j]*(c1**-self.sigma)))
            err_n = (w0*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)
                     *(1-n0**self.ellip_up)**(1/self.ellip_up-1))
            err = np.array([err_c, err_n])
            return err
        else:
            c0 = self.get_c0(b1n0, b0, s, j, i, w, r)
            c1 = self.get_c1(b1n0, b0, s, j, i, w, r)

            err_c = (c0**-self.sigma
                     -self.beta*(1+r)*np.sum(self.Pi[j]*(c1**-self.sigma)))
            err_n = (w*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)
                     *(1-n0**self.ellip_up)**(1/self.ellip_up-1))
            err = np.array([err_c, err_n])
            return err
   

    def obj2(self, b1n0, b0, s, j, i, w, r):
        """Calculate the euler errors and take their norm."""
        b1, n0 = b1n0
        if isinstance(w, (np.ndarray, list)):
            w0, w1 = w
            r0, r1 = r
            c0 = self.get_c0(b1n0, b0, s, j, i, w0, r0)
            c1 = self.get_c1(b1n0, b0, s, j, i, w1, r1)

            err_c = (c0**-self.sigma
                     -self.beta*(1+r1)*np.sum(self.Pi[j]*(c1**-self.sigma)))
            err_n = (w0*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)
                     *(1-n0**self.ellip_up)**(1/self.ellip_up-1))
            err = np.array([err_c, err_n])
            return np.linalg.norm(err)/1000000.
        else:
            c0 = self.get_c0(b1n0, b0, s, j, i, w, r)
            c1 = self.get_c1(b1n0, b0, s, j, i, w, r)

            err_c = (c0**-self.sigma
                     -self.beta*(1+r)*np.sum(self.Pi[j]*(c1**-self.sigma)))
            err_n = (w*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)
                     *(1-n0**self.ellip_up)**(1/self.ellip_up-1))
            err = np.array([err_c, err_n])
            return np.linalg.norm(err)/1000000.
            

    def update_Psi_n_S(self, w, r):
        """Update the (S,j) rows of the labor policy function."""
        for j in xrange(self.J):
            for i in range(self.grid_size):
                grid = self.Grid[-1,j,i].copy()
                
                def get_n_S(n):
                    """Calculate the Euler errors."""
                    return (w*self.e[j]
                            *(w*self.e[j]*n+grid*(1+r))**-sigma
                           -(self.ellip_b*n**(self.ellip_up-1)
                            *(1-n**self.ellip_up)**(1/self.ellip_up-1)))
                
                lb = max(0, -grid*(1+r)/(w*self.e[j])+.000000001)
                ub = 1.
                
                n_S, res = brentq(get_n_S, lb, ub, full_output=1)
                self.euler_errors.append((self.S-1,j,i))
                self.euler_errors.append(get_n_S(n_S))
                
                if res.converged!=1:
                    print 'n_S did not converge!'
                
                self.Psi_n[-1,j,i] = n_S  
    
        
    def update_Psi_b1n0(self, s, j, w, r):
        """Update the (s,j) of the labor and savings policy functions."""
        if isinstance(w, (np.ndarray, list)): # This means we are doing TPI.
            # These are cases where TPI is trying to solve unnecessary problems
            # so we skip them. They are marked by w=999.
            if w[0]==999.: 
                self.Psi_b[s,j], self.Psi_n[s,j] = 0.0, 0.0
                return ' '
            # When calculating TPI these cases are already solved so we don't
            # recalculate them.
            elif w[0]==self.wSS:   
                self.Psi_b[s,j] = self.Psi_bSS[s,j]
                self.Psi_n[s,j] = self.Psi_nSS[s,j]
                return None
            else:
                # First we solve for an easy problem so we can use its
                # results as an initial guess for other problems.
                i = self.grid_size/2
                # print s, j, i

                # Make grid of initial wealths
                self.Grid[s,j] = np.linspace(self.b0min,self.b0max,self.grid_size)

                obj = lambda x: self.obj2(x, self.Grid[s,j,i], s, j, i, w, r) 
                # Calculate a feasible guess
                if s == self.S-2:
                    guess = [w[0]*self.e[j]*.8+self.b0min*(1+r), .9]

                else:
                    guess = self.b1n0i
                # Define constraints and bounds
                cons = ({'type': 'ineq',
                         'fun': lambda x: self.get_c0(x, self.Grid[s,j,i],
                                                      s, j, i, w[0], r[0])},
                        {'type': 'ineq',
                         'fun': lambda x: self.get_c1(x, self.Grid[s,j,i],
                                                      s, j, i, w[1], r[1])[0]},
                        {'type': 'ineq',
                         'fun': lambda x: self.get_c1(x, self.Grid[s,j,i],
                                                      s, j, i, w[1], r[1])[1]})

                bnds = ((None,None),(0,1))
                # Solve using a constrained minimizer
                sol = minimize(obj, guess, bounds=bnds, constraints=cons, tol=1e-15)
                b1n0, ier, fun = sol.x, sol.success, sol.fun

                self.euler_errors.append(str((s,j,i)))
                if ier==1:
                    self.euler_errors.append(fun)
                else:
                    self.euler_errors.append(999.)
                self.b1n0i = b1n0
                # Update the appropriate entrys of the policy function arrays.
                (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0

                if ier!=1:
                    print "Couldn't solve for the agent's choices"
                    print "s, j, i: ", s, j, i
                    print "Error message: ", sol.message
                    print "Initial guess when this happened ", guess

                # TODO: The  next two chunks could be done in parallel,
                # working out from the middle.

                # Solve the problems with initial wealth lower.
                for i in xrange(i-1, -1, -1):
                    self.euler_errors.append(str((s,j,i)))
                    obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i, w, r)
                    # Set the initial guess.
                    if i==self.grid_size/2-1:
                        guess = self.b1n0i
                    else:
                        guess = b1n0
                    # Solve.
                    b1n0, info, ier, mesg = fsolve(obj, guess, full_output=1)
                    if ier==1:
                        self.euler_errors.append(info['fvec'])
                    # Update Psi_b and Psi_n
                    (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0

                    if ier!=1:
                        print "Couldn't solve for the agent's choices."
                        print "s, j, i: ", s, j, i
                        print "Error message: ", mesg
                        print "Initial guess when this happened ", guess
                        print "Trying to solve again..."
                        guess = b1n0
                        # Try to solve again...
                        b1n0, info, ier, mesg = fsolve(obj, guess, full_output=1)
                        if ier==1:
                            self.euler_errors.append(info['fvec'])
                        else:
                            self.euler_errors.append(999)                    
                        (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0
                        print i, b1n0, obj(b1n0)
                        if ier != 1:
                            print "Failed again."
                            print "s, j, i: ", s, j, i
                            print "Error message: ", mesg
                            print "Trying again with a higher b0min"
                            print "old b0min ", self.b0min
                            print "new b0min ", self.Grid[s,j,i+1]
                            self.b0min = self.Grid[s,j,i+1]
                            self.update_Psi_b1n0(s, j, w, r)
                            return None

                # Solve the problems with initial wealth higher.       
                for i in xrange(self.grid_size/2+1, self.grid_size):
                    self.euler_errors.append(str((s,j,i)))
                    obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i, w, r)
                    # Set inital guess
                    if i==self.grid_size/2+1:
                        guess = self.b1n0i
                    else:
                        guess = b1n0
                    # Solve
                    b1n0, info, ier, mesg = fsolve(obj, guess, full_output=1)
                    if ier==1:
                        self.euler_errors.append(info['fvec'])
                    else:
                        self.euler_errors.append(999)
                    # Update
                    (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0            
                    if ier!=1:
                        print "Couldn't solve for the agent's choices."
                        print "s, j, i: ", s, j, i
                        print "Error message: ", mesg
            
        else: # This means we are calculating SS.
            # First we solve for an easy problem so we can use its
            # results as an initial guess for other problems.
            i = self.grid_size/2
            # print s, j, i

            # Make grid of initial wealths
            self.Grid[s,j] = np.linspace(self.b0min,self.b0max,self.grid_size)

            obj = lambda x: self.obj2(x, self.Grid[s,j,i], s, j, i, w, r) 
            # Calculate a feasible guess
            if s == self.S-2:

                guess = [w*self.e[j]*.8+self.b0min*(1+r), .9]

            else:
                guess = self.b1n0i
            # Define constraints and bounds
            cons = ({'type': 'ineq',
                     'fun': lambda x: self.get_c0(x, self.Grid[s,j,i],
                                                  s, j, i, w, r)},
                    {'type': 'ineq',
                     'fun': lambda x: self.get_c1(x, self.Grid[s,j,i],
                                                  s, j, i, w, r)[0]},
                    {'type': 'ineq',
                     'fun': lambda x: self.get_c1(x, self.Grid[s,j,i],
                                                  s, j, i, w, r)[1]})
            bnds = ((None,None),(0,1))
            # Solve using a constrained minimizer
            sol = minimize(obj, guess, bounds=bnds, constraints=cons, tol=1e-15)
            b1n0, ier, fun = sol.x, sol.success, sol.fun

            self.euler_errors.append(str((s,j,i)))
            if ier==1:
                self.euler_errors.append(fun)
            else:
                self.euler_errors.append(999.)
            self.b1n0i = b1n0
            # Update the appropriate entrys of the policy function arrays.
            (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0

            if ier!=1:
                print "Couldn't solve for the agent's choices"
                print "s, j, i: ", s, j, i
                print "Error message: ", sol.message
                print "Initial guess when this happened ", guess

            # TODO: The  next two chunks could be done in parallel, working out from the middle.

            # Solve the problems with initial wealth lower.
            for i in xrange(i-1, -1, -1):
                self.euler_errors.append(str((s,j,i)))
                obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i, w, r)
                # Set the initial guess.
                if i==self.grid_size/2-1:
                    guess = self.b1n0i
                else:
                    guess = b1n0
                # Solve.
                b1n0, info, ier, mesg = fsolve(obj, guess, full_output=1)
                if ier==1:
                    self.euler_errors.append(info['fvec'])
                # Update Psi_b and Psi_n
                (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0

                if ier!=1:
                    print "Couldn't solve for the agent's choices."
                    print "s, j, i: ", s, j, i
                    print "Error message: ", mesg
                    print "Initial guess when this happened ", guess
                    print "Trying to solve again..."
                    guess = b1n0
                    # Try to solve again...
                    b1n0, info, ier, mesg = fsolve(obj, guess, full_output=1)
                    if ier==1:
                        self.euler_errors.append(info['fvec'])
                    else:
                        self.euler_errors.append(999)                    
                    (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0
                    print i, b1n0, obj(b1n0)
                    if ier != 1:
                        print "Failed again."
                        print "s, j, i: ", s, j, i
                        print "Error message: ", mesg
                        print "Trying again with a higher b0min"
                        print "old b0min ", self.b0min
                        print "new b0min ", self.Grid[s,j,i+1]
                        self.b0min = self.Grid[s,j,i+1]
                        self.update_Psi_b1n0(s, j, w, r)
                        return None

            # Solve the problems with initial wealth higher.       
            for i in xrange(self.grid_size/2+1, self.grid_size):
                self.euler_errors.append(str((s,j,i)))
                obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i, w, r)
                # Set inital guess
                if i==self.grid_size/2+1:
                    guess = self.b1n0i
                else:
                    guess = b1n0
                # Solve
                b1n0, info, ier, mesg = fsolve(obj, guess, full_output=1)
                if ier==1:
                    self.euler_errors.append(info['fvec'])
                else:
                    self.euler_errors.append(999)
                # Update
                (self.Psi_b[s,j,i], self.Psi_n[s,j,i]) = b1n0            
                if ier!=1:
                    print "Couldn't solve for the agent's choices."
                    print "s, j, i: ", s, j, i
                    print "Error message: ", mesg


    def update_Psi(self, w_path=None, r_path=None):
        """Update the entirety of both policy functions."""
        if w_path==None:
            self.euler_errors = []
            self.update_Psi_n_S(self.w, self.r)
            for s in xrange(self.S-2,-1,-1):
                for j in xrange(self.J):
                    self.update_Psi_b1n0(s, j, self.w, self.r)
        else:
            self.update_Psi_n_S(w_path[-1], r_path[-1])
            for s in xrange(self.S-2,-1,-1):
                for j in xrange(self.J):
                    print "s, j:", s, j
                    self.update_Psi_b1n0(s, j, w_path[s:s+2], r_path[s:s+2])
    
    
    def update_B_and_N(self):
        """Given the policy functions, update the states B and N"""
        for s in range(self.S-1):
            for j in range(self.J):
                psi_b = UnivariateSpline(self.Grid[s,j], self.Psi_b[s,j])
                psi_n = InterpolatedUnivariateSpline(self.Grid[s,j],
                                                     self.Psi_n[s,j])
                self.B[s+1,
                       self.shocks[s]==j] = psi_b(self.B[s,
                                                         self.shocks[s]==j])
                self.N[s,
                       self.shocks[s]==j] = psi_n(self.B[s,
                                                         self.shocks[s]==j])

        for j in range(self.J):
            psi_n = InterpolatedUnivariateSpline(self.Grid[S-1,j],
                                                 self.Psi_n[S-1,j])
            self.N[S-1,
                   self.shocks[S-1]==j] = psi_n(self.B[S-1,
                                                       self.shocks[S-1]==j])


    def set_state(self):
        """Given B and N determine L, K, r, and w."""
        self.L = np.sum(self.N)
        self.K = np.sum(self.B)

        if self.K<0:
            print "K is negative!"
        
        self.r = self.alpha*self.A*((self.L/self.K)**(1-self.alpha))-self.delta
        self.w = (1-self.alpha)*self.A*((self.K/self.L)**self.alpha)

        
    def calc_SS(self, tol=1e-4, maxiter=10000):
        """Calculate the steady state."""
        self.r_w_ = []
        self.start = time.time()
        diff = 1
        count = 0
        while diff>tol and count<maxiter:
            r0, w0 = self.r, self.w
            self.update_Psi()
            self.update_B_and_N()
            self.set_state()
            print 'r and w', self.r, self.w
            print 'max of B', np.max(self.B)
            print 'min of B', np.min(self.B)
            count += 1
            diff = max(np.abs(self.r-r0), np.abs(self.w-w0))
            self.r, self.w = .2*self.r+.8*r0, .2*self.w+.8*w0
            self.r_w_.append(self.r)
            self.r_w_.append(self.w)
            print count, diff
        self.rSS = self.r
        self.wSS = self.w
        self.KSS = self.K
        self.LSS = self.L
        self.BSS = self.B
        self.NSS = self.N
        self.Psi_bSS = self.Psi_b
        self.Psi_nSS = self.Psi_n
        self.write_output()
        
        
    def calc_TPI(self, b0, lambda_0, T=10, tol=10e-4, maxiter=100):
        """Calculate the time path."""
        self.b0min = self.B.min()
        M = self.M*self.S

        # Make shocks
        initial_e = np.zeros(M).astype('int')
        for agent in M*lambda_0[:-1]:
            initial_e[agent:] += 1
        shocks = self.MC.simulate(T, initial_e, random_state=1)

        # Make paths
        K0 = b0.sum(0)
        self.K_path = np.linspace(K0, self.KSS, T)
        self.L_path = np.ones(T)*self.LSS

        # Initialize B and N
        B = np.empty((M, T))
        B[:,0] = b0
        B[:,-1] = self.BSS.flatten()
        N = np.empty((M, T))
        
        diff = 1
        count = 0
        while diff>tol and count<maxiter:
            r_path = np.ones(T+2*self.S-2)*999
            w_path = np.ones(T+2*self.S-2)*999
            r_path0 = self.alpha*self.A*((self.L_path/self.K_path)**(1-self.alpha))-self.delta
            w_path0 = (1-self.alpha)*self.A*((self.K_path/self.L_path)**self.alpha)
            r_path[self.S-1:self.S+T-1] = r_path0
            w_path[self.S-1:self.S+T-1] = w_path0
            r_path[-self.S+1:] = self.rSS
            w_path[-self.S+1:] = self.wSS
            print r_path
            print w_path

            for s in range(self.S):
                print "s", s
                B_ = B[s*self.M:(s+1)*self.M]
                N_ = N[s*self.M:(s+1)*self.M]
                shocks_ = shocks[s*self.M:(s+1)*self.M]
            
                num = 1
                if s > self.S-T:
                    num = 2

                for n in range(num):
                    w_path_ = w_path[self.S-s+n*self.S-1:self.S-s+(n+1)*self.S-1]
                    r_path_ = r_path[self.S-s+n*self.S-1:self.S-s+(n+1)*self.S-1]
                    print w_path_
                    print r_path_

                    # Make the policy functions and update B and N
                    self.b0min = -1.3
                    self.update_Psi(w_path_, r_path_)
                
                    if n == 0:
                        start = 0
                        end = T - (num-1)*(s-self.S+T)

                        for t in range(start, end-1):
                            for j in range(self.J):
                                psi_b = UnivariateSpline(self.Grid[t+s,j],
                                                         self.Psi_b[t+s,j])
                                psi_n = InterpolatedUnivariateSpline(self.Grid[t+s,j],
                                                                     self.Psi_n[t+s,j])
                                B_[:,t+1][shocks_[:,t]==j] = psi_b(B_[:,t][shocks_[:,t]==j])
                                N_[:,t][shocks_[:,t]==j] = psi_n(B_[:,t][shocks_[:,t]==j])
                        for j in range(self.J):
                            psi_n = InterpolatedUnivariateSpline(self.Grid[t++s+1,j],
                                                                 self.Psi_n[t+s+1,j])
                            N_[:,t+1][shocks_[:,t+1]==j] = psi_n(B_[:,t+1][shocks_[:,t+1]==j])
    
                    if n == 1:
                        start = self.S-s
                        end = T
                        for t in range(start, end-1):
                            for j in range(self.J):
                                psi_b = UnivariateSpline(self.Grid[t-start,j],
                                                         self.Psi_b[t-start,j])
                                psi_n = InterpolatedUnivariateSpline(self.Grid[t-start,j],
                                                                     self.Psi_n[t-start,j])
                                B_[:,t+1][shocks_[:,t]==j] = psi_b(B_[:,t][shocks_[:,t]==j])
                                N_[:,t][shocks_[:,t]==j] = psi_n(B_[:,t][shocks_[:,t]==j])
                        for j in range(self.J):
                            psi_n = InterpolatedUnivariateSpline(self.Grid[t-start+1,j],
                                                                 self.Psi_n[t-start+1,j])
                            N_[:,t+1][shocks_[:,t+1]==j] = psi_n(B_[:,t+1][shocks_[:,t+1]==j])
                            

            K_path1 = B.sum(0)
            L_path1 = N.sum(0)
            r_path1 = self.alpha*self.A*((L_path1/K_path1)**(1-self.alpha))-self.delta
            w_path1 = (1-self.alpha)*self.A*((K_path1/L_path1)**self.alpha)
            diff = np.max(np.linalg.norm(r_path0-r_path1), np.linalg.norm(w_path0-w_path1))
            count += 1
            print count, diff 
            self.K_path = .2*K_path1+.8*self.K_path
            self.L_path = .2*L_path1+.8*self.L_path
        
        
    def write_output(self):
        """
        Writes ouputfile to 'Output/output.txt'

        Example:
        Date Time
        Total SS computation time:
        Total Labor:
        Total Capital:
        SS interest rate:
        SS wage:
        """
        self.date = time.strftime("%m/%d/%Y")
        self.time = time.strftime("%I:%M:%S %p")
        total_time = str(datetime.timedelta(seconds=time.time() - self.start))
        date = str(datetime.datetime.now())
        output = open('Output/%s.txt' % date, 'w')
        output.write('{} {}\n'.format(self.date, self.time))
        output.write('Total SS computation time: {}\n'.format(total_time))
        output.write('Total Labor supply: {}\
                     \nTotal Capital supply: {}\
                     \nSteady State interest rate: {}\
                     \nSteady State Wage: {}\
                     \n'.format(self.L, self.K, self.r, self.w))
        output.close()


# An example of an OG object:

Num = 1000
S = 80
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.4, 0.6],
               [0.6, 0.4]])
e = np.array([0.8, 1.2])
theta = 2.
household_params = (Num, S, J, beta_annual, sigma, Pi, e, theta)

A = 1.0
alpha = .35
delta_annual = .05
firm_params = (A, alpha, delta_annual)


# Another example, this one with three ability types:

# Num = 1000
# S = 80
# J = 3
# beta_annual = .96
# sigma = 3.0
# Pi = np.array([[0.4, 0.4, 0.2],
#                [0.3, 0.4, 0.3],
#                [0.2, 0.4, 0.4]])
# e = np.array([0.8, 1.0, 1.2])
# theta = 2.
# household_params = (Num, S, J, beta_annual, sigma, Pi, e, theta)

# A = 1.0
# alpha = .35
# delta_annual = .05
# firm_params = (A, alpha, delta_annual)

# Initialize the OG object.
og = OG(household_params, firm_params)

# Calculate the steady state and a time path.
# og.calc_SS()
# b0 = og.BSS.flatten()*.9
# lambda0 = og.lambda_bar*.9
# og.calc_TPI(b0, lambda0)

