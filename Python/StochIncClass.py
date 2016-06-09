import numpy as np
from scipy.optimize import fsolve, brentq, root, minimize
from quantecon import markov
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from ellipse import estimation
import time
import datetime

class OG(object):
    """OG object that solves overlapping generations problems.
    
    ---Methods---
    update_Psi: Updates the policy functions
    jac: Calculates the jacobian at a specific point
    obj: Calculates Euler errors for capital and labor
    update_B_and_N: Updates the total capital and labor objects
    set_state: resets state variables
    calc_SS: iterates until convergence is reached
    """
    def __init__(self, household_params, firm_params):
        """Instantiate the state parameters of the OG model.

        Attributes
        ----------
        ### Household params ###
        Num: int, number of agents in smallest ability group
        S: int, number of time periods in agent life
        J: int, number of ability groups
        beta: float, discount factor
        sigma: float, constant of relative risk aversion
        Pi: (J,J) array, Probability of movement between ability groups
        e: (J,) array, effective labor units given ability type
        lambda_bar: (J,) array, steady state ergodic distribution ability
                                                                 of agents
        agents: (J,) array, number of agents in each ability group
        M: int, total number of agents
        shocks: (S,M) agents, ability life path of all agents
        theta: float, agent elliptical utility labor param

        ### Firm Params ###
        A: float, firm production multiplier
        alpha: float, cobb-douglas production parameter
        delta: float, depreciation

        ### State Params ###
        B: (S,M) array, Savings decision for M agents, S periods
        N: (S,M) array, Labor decisions for M agents, S periods
        r: float, interest rate
        w: float, wage rate
        grid_size: int, number of points in policy function grid
        b0min: float, min endpoint for policy function interpolation
        b0max: float, max endpoint for policy function interpolation
        Grid: (S,J,grid_size) array, policy grids for period s, ability j
        Psi_b: (S,J,grid_size) array, Capital policy function
                                                     for period s, ability j
        Psi_n: (S,J,grid_size) array, Labor policy function
                                                    for period s, ability j
        ellip_up: float, estimated upsilon from elliptical utility package
        ellip_b: float, estimated b from elliptical utility package
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
        MC = markov.core.MarkovChain(self.Pi)
        self.lambda_bar = MC.stationary_distributions
        weights = (self.lambda_bar/(self.lambda_bar.min())*self.Num).astype('int')
        self.M = np.sum(weights)
        initial_e = np.zeros(self.M)
        for agent in np.cumsum(weights[0][1:]):
            initial_e[agent:] += 1
        self.shocks = MC.simulate(self.S, initial_e.astype('int'), random_state=1).T
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
        self.r, self.w = 0.05137839, 1.2667007 # May need to have good guesses.
        
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
        
        
    def get_c0(self, b1n0, b0, s, j, i):
        """Calculate consumption."""
        b1, n0 = b1n0
        return b0*(1+self.r)+self.w*self.e[j]*n0-b1
    
    def get_c1(self, b1n0, b0, s, j, i):
        """Calculate consumption."""
        b1, n0 = b1n0
        b2 = np.array(
             [UnivariateSpline(self.Grid[s+1,j_],
                               self.Psi_b[s+1,j_])(b1) for j_ in xrange(self.J)]
        )
        n1 = np.array(
             [UnivariateSpline(self.Grid[s+1,j_],
                               self.Psi_n[s+1,j_])(b1) for j_ in xrange(self.J)])
        return b1*(1+self.r)+self.w*self.e*n1-b2
    
    def obj(self, b1n0, b0, s, j, i):
        """Calculate the euler errors."""
        b1, n0 = b1n0
       
        c0 = self.get_c0(b1n0, b0, s, j, i)
        c1 = self.get_c1(b1n0, b0, s, j, i)
    
        err_c = (c0**-self.sigma
                 -self.beta*(1+self.r)*np.sum(self.Pi[j]*(c1**-self.sigma)))
        err_n = (self.w*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)
                 *(1-n0**self.ellip_up)**(1/self.ellip_up-1))
        err = np.array([err_c, err_n])
        return err
   

    def obj2(self, b1n0, b0, s, j, i):
        """Calculate the euler errors and take their norm."""
        b1, n0 = b1n0
       
        c0 = self.get_c0(b1n0, b0, s, j, i)
        c1 = self.get_c1(b1n0, b0, s, j, i)
    
        err_c = (c0**-self.sigma
                 -self.beta*(1+self.r)*np.sum(self.Pi[j]*(c1**-self.sigma)))
        err_n = (self.w*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)
                 *(1-n0**self.ellip_up)**(1/self.ellip_up-1))
        err = np.array([err_c, err_n])
        return np.linalg.norm(err)/1000000.

        
    def update_Psi_n_S(self):
        """Update the (S,j) rows of the labor policy function."""
        for j in xrange(self.J):
            for i in range(self.grid_size):
                grid = self.Grid[-1,j,i].copy()
                
                def get_n_S(n):
                    """Calculate the Euler errors."""
                    return (self.w*self.e[j]
                            *(self.w*self.e[j]*n+grid*(1+self.r))**-sigma
                           -(self.ellip_b*n**(self.ellip_up-1)
                            *(1-n**self.ellip_up)**(1/self.ellip_up-1)))
                
                lb = max(0, -grid*(1+self.r)/(self.w*self.e[j])+.000000001)
                ub = 1.
                
                n_S, r = brentq(get_n_S, lb, ub, full_output=1)
                self.euler_errors.append((self.S-1,j,i))
                self.euler_errors.append(get_n_S(n_S))
                
                if r.converged!=1:
                    print 'n_S did not converge!'
                
                self.Psi_n[-1,j,i] = n_S  
                
                
    def update_Psi_b1n0(self, s, j):
        """Update the (s,j) of the labor and savings policy functions."""
        # First we solve for an easy problem so we can use its
        # results as an initial guess for other problems.
        i = self.grid_size/2
        print s, j, i
        # Make grid of initial wealths
        self.Grid[s,j] = np.linspace(self.b0min,self.b0max,self.grid_size)
        
        obj = lambda x: self.obj2(x, self.Grid[s,j,i], s, j, i) 
        # Calculate a feasible guess
        if s == self.S-2:
            guess = [self.w*self.e[j]*.8+self.b0min*(1+self.r), .9]
        else:
            guess = self.b1n0i
        # Define constraints and bounds
        cons = ({'type': 'ineq',
                 'fun': lambda x: self.get_c0(x, self.Grid[s,j,i],
                                              s, j, i)},
                {'type': 'ineq',
                 'fun': lambda x: self.get_c1(x, self.Grid[s,j,i],
                                              s, j, i)[0]},
                {'type': 'ineq',
                 'fun': lambda x: self.get_c1(x, self.Grid[s,j,i],
                                              s, j, i)[1]})
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
            print s, j, i, sol.message
            print guess
        
        # TODO: The  next two chunks could be done in parallel, working out from the middle.
        
        # Solve the problems with initial wealth lower.
        for i in xrange(i-1, -1, -1):
            self.euler_errors.append(str((s,j,i)))
            obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i)
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
                print s, j, i, mesg
                print guess
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
                    print '!!!!!!!!!!', s, j, i, mesg
                    print self.b0min, self.Grid[s,j,i+1]
                    self.b0min = self.Grid[s,j,i+1]
                    self.update_Psi_b1n0(s, j)
                    return None
                
        # Solve the problems with initial wealth higher.       
        for i in xrange(self.grid_size/2+1, self.grid_size):
            self.euler_errors.append(str((s,j,i)))
            obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i)
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
                print s, j, i, mesg
    

    def update_Psi(self):
        """Update the entirety of both policy functions."""
        self.euler_errors = []
        self.update_Psi_n_S()
        for s in xrange(self.S-2,-1,-1):
            for j in xrange(self.J):
                self.update_Psi_b1n0(s, j)
    
    
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

        
    def calc_SS(self, tol=1e-5, maxiter=10000):
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
        self.write_output()

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
        output = open('Output/output.txt', 'w')
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

og = OG(household_params, firm_params)
