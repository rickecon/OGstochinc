import numpy as np
from scipy.optimize import fsolve, root, minimize
from quantecon import markov
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
import matplotlib.pyplot as plt

class OG(object):
    """
    """
    def __init__(self, household_params, firm_params):
        """Instantiate the state parameters of the OG model."""
        (self.N, self.S, self.J, beta_annual, self.sigma,
         self.Pi, self.e) = household_params
        
        self.beta = beta_annual**(80/self.S)

        n = np.ones(self.S)
        n[2*S/3:] = n[2*S/3:]*.3
        self.n = n
        
        MC = markov.core.MarkovChain(self.Pi)
        self.lambda_bar = MC.stationary_distributions
        
        weights = (self.lambda_bar/(self.lambda_bar.min())*self.N).astype('int')
        
        self.M = np.sum(weights)
        
        initial_e = np.zeros(self.M)
        for weight in np.cumsum(weights[0][1:]):
            initial_e[weight:] += 1
        
        self.shocks = MC.simulate(self.S, initial_e, random_state=None).T
        self.abilities = self.e[self.shocks]

        (self.A, self.alpha, delta_annual) = firm_params
        
        self.delta = 1-(1-delta_annual)**(80/self.S)
        
        self.B = np.empty((self.S,self.M))
        self.B[0] = 0.0
        
        self.grid_size = 101
        self.grid = np.linspace(-.733333, 7, self.grid_size)
        self.Psi = np.empty((self.S, self.J, self.grid_size))
        self.Psi[-1] = 0.0
        
    
    def update_Psi(self):
        for j in range(self.J):
            for s in range(self.S-2,-1,-1):
                psi = UnivariateSpline(self.grid, self.Psi[s+1,j])
                lb = -self.w*np.min(self.e)*self.n[s+1]/(1+self.r)+psi(self.grid)
                ub = self.grid*(1+self.r)+self.w*self.e[j]*self.n[s]
                guess = (lb+ub)/2.
                self.Psi[s,j], info, ier, mesg = fsolve(self.obj, guess, args=(self.grid, psi, s, j), full_output=1)
                if ier!=1:
                    print 'For age {s} and ability {j} we didn\'t converge.'.format(**locals())
                
    
    def obj(self, b1, b0, psi, s, j):
        b2 = psi(b1)
        c0 = b0*(1+self.r)+self.w*self.e[j]*self.n[s]-b1
        c1 = b1[:,None]*(1+self.r)+self.w*self.e*self.n[s+1]-b2[:,None]
        err = c0**-self.sigma-self.beta*(1+self.r)*np.inner(self.Pi[j],c1**-self.sigma)
        return err

    
    def update_B(self):
        for s in range(self.S-1):
            for j in range(self.J):
                psi = InterpolatedUnivariateSpline(self.grid, self.Psi[s,j])
                self.B[s+1, self.shocks[s]==j] = psi(self.B[s, self.shocks[s]==j])
                
            
    def set_state(self):
        self.L = self.n.sum()*self.M
        weightedK = np.array([self.B[self.shocks==j].sum() for j in range(self.J)])
        self.K = np.sum(self.lambda_bar*weightedK)/self.S

        self.r = self.alpha*self.A*((self.L/self.K)**(1-self.alpha))-self.delta
        self.w = (1-self.alpha)*self.A*((self.K/self.L)**self.alpha)
        
        b0min = -self.w*(2+self.r)/(1+self.r)**2*np.min(self.e)*np.min(self.n)
        self.grid = np.linspace(np.round(b0min,5), 7, self.grid_size)

    
    def calc_SS(self, tol=1e-10, maxiter=100):
        self.r, self.w = 3., .2
        diff = 1
        count = 0
        while diff>tol and count<maxiter:
            r0, w0 = self.r, self.w
            B0 = self.B
            self.update_Psi()
            self.update_B()
            self.set_state()
            count += 1
            diff = max(np.abs(self.r-r0), np.abs(self.w-w0))
            print count, diff

N = 500
S = 10
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
og.calc_SS()
