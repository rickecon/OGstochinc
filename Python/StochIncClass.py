import numpy as np
from scipy.optimize import fsolve
from quantecon import markov
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
import matplotlib.pyplot as plt

class OG(object):
    def __init__(self, household_params, firm_params):
        """Instantiate the state parameters of the OG model."""
        # Make household parameters as attributes.
        (self.N, self.S, self.J, beta_annual, self.sigma,
         self.Pi, self.e) = household_params
        # Convert beta_annual to the beta for a period.
        self.beta = beta_annual**(80/self.S)
        # Exogenous labor.
        n = np.ones(self.S)
        n[2*self.S/3:] = n[2*self.S/3:]*.3
        self.n = n
        # Create the markov chain object so we can simulate the shock paths.
        MC = markov.core.MarkovChain(self.Pi)
        # Find the ergodic distribution.
        self.lambda_bar = MC.stationary_distributions
        # Determine the number of agents initially in each ability type.
        weights = (self.lambda_bar/(self.lambda_bar.min())*self.N).astype('int')
        self.M = np.sum(weights)
        initial_e = np.zeros(self.M)
        for weight in np.cumsum(weights[0][1:]):
            initial_e[weight:] += 1
        # Simulate M shock paths.
        self.shocks = MC.simulate(self.S, initial_e, random_state=1).T
        self.abilities = self.e[self.shocks]
        # Set the firm parameters as attributes.
        (self.A, self.alpha, delta_annual) = firm_params
        # Convert delta_annual to the delta corresponding to the period length.
        self.delta = 1-(1-delta_annual)**(80/self.S)
        # Initialize the state.
        self.B = np.empty((self.S,self.M))
        self.B[0] = 0.0
        # Set initial guesses for r and w.
        self.r, self.w = 0.312381791072, 0.637945371028
        # Initialize
        self.grid_size = 3500
        self.b0min = -self.w*(2+self.r)/(1+self.r)**2*np.min(self.e)*np.min(self.n)+1e-5
        self.b0max = 1500
        self.Grid = np.empty((self.S, self.J, self.grid_size))
        for j in range(self.J):
            self.Grid[-1,j] = np.linspace(self.b0min, self.b0max, self.grid_size)
        self.Psi = np.empty((self.S, self.J, self.grid_size))
        self.Psi[-1] = 0.0
        
                
    def update_Psi(self):
        for s in range(self.S-2,-1,-1):
            for j in range(self.J):
                lb = -999
                for j_ in range(self.J):
                    psi = UnivariateSpline(self.Grid[s+1,j_], self.Psi[s+1,j_])                
                    def get_lb(b1):
                        b2 = psi(b1)
                        c1 = b1*(1+self.r)+self.w*self.e[j_]*self.n[s+1]-b2
                        return c1
                    guess = (-self.w*self.e[j_]*self.n[s+1]+psi(0))/(1+self.r)
                    lb_, info, ier, mesg = fsolve(get_lb, guess, full_output=1)
                    lb = np.max([lb,lb_])
                    if ier!=1:
                        print s, j, j_, 'The lower bound wasn\'t calculated correctly.'
                self.lb = lb
                self.b0min = (lb-self.w*self.e[j]*self.n[s])/(1+self.r)+1e-5
                self.Grid[s,j] = np.linspace(self.b0min, self.b0max, self.grid_size)
                ub = self.Grid[s,j]*(1+self.r)+self.w*self.e[j]*self.n[s]
                self.ub = ub
                psi = UnivariateSpline(self.Grid[s+1,j], self.Psi[s+1,j])
                print s, j, ub[0]-lb
                for i in range(self.grid_size):
                    obj = lambda x: self.obj(x, self.Grid[s,j,i], psi, s, j, i)
                    self.Psi[s,j,i], info, ier, mesg = fsolve(obj, (lb+ub[i])/2., full_output=1)
                    print obj()
                    if ier!=1:
                        print s, j, i, 'no converge'
                            
    
    def obj(self, b1, b0, psi, s, j, i):
        if b1<self.lb:
            return -np.inf-np.abs(b1)
        if b1>self.ub[i]:
            return np.inf+np.abs(b1)
        b2 = psi(b1)
        c0 = b0*(1+self.r)+self.w*self.e[j]*self.n[s]-b1
        c1 = b1*(1+self.r)+self.w*self.e*self.n[s+1]-b2
        err = c0**-self.sigma-self.beta*(1+self.r)*np.sum(self.Pi[j]*(c1**-self.sigma))
        return err

    
    def update_B(self):
        for s in range(self.S-1):
            for j in range(self.J):
                psi = UnivariateSpline(self.Grid[s,j], self.Psi[s,j])
                self.B[s+1, self.shocks[s]==j] = psi(self.B[s, self.shocks[s]==j])
                
            
    def set_state(self):
        self.L = self.n.sum()*self.M
        weightedK = np.array([self.B[self.shocks==j].sum() for j in range(self.J)])
        self.K = np.sum(self.lambda_bar*weightedK)/self.S

        self.r = self.alpha*self.A*((self.L/self.K)**(1-self.alpha))-self.delta
        self.w = (1-self.alpha)*self.A*((self.K/self.L)**self.alpha)
        
        self.b0min = -self.w*(2+self.r)/(1+self.r)**2*np.min(self.e)*np.min(self.n)+1e-5
        for j in range(self.J):
            self.Grid[-1,j] = np.linspace(self.b0min, self.b0max, self.grid_size)

    
    def calc_SS(self, tol=1e-10, maxiter=100):
        diff = 1
        count = 0
        while diff>tol and count<maxiter:
            r0, w0 = self.r, self.w
            B0 = self.B
            self.update_Psi()
            self.update_B()
            self.set_state()
            print 'r and w', self.r, self.w
            print 'max of B', np.max(og.B)
            count += 1
            diff = max(np.abs(self.r-r0), np.abs(self.w-w0))
            self.r, self.w = .2*self.r+.8*r0, .2*self.w+.8*w0
            print count, diff

N = 1000
S = 80
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.4, 0.6],
               [0.6, 0.4]])
e = np.array([0.8, 1.2])

household_params = (N, S, J, beta_annual, sigma, Pi, e)

A = 1.0
alpha = .35
delta_annual = .05
firm_params = (A, alpha, delta_annual)

og = OG(household_params, firm_params)