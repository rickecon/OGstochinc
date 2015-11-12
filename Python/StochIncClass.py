import time

import numpy as np
from scipy.optimize import fsolve

class OG(object):
    """
    
    """
    def __init__(self, household_params, firm_params):
        """Instatiate the state parameters of the OG model."""
        (self.N,
         self.S,
         self.J,
         self.beta_annual,
         self.sigma,
         self.Pi,
         self.e_jt,
         self.mean,
         self.std) = household_params
        self.beta = self.beta_annual**(80/self.S)

        nvec = np.ones(S)
        nvec[2*S/3:] = nvec[2*S/3:]*.3
        self.nvec = nvec
        self.lambda_bar = self.get_lambda_bar(self.Pi)

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params
        self.delta = 1-(1-self.delta_annual)**(80/self.S)
    

    def set_state(self):
        """Set initial state and r and w."""
        self.initialize_b_vec()
        self.get_r_and_w()
        #self.get_c()


    def get_lambda_bar(self, Pi):
        """Compute the ergodic distribution of the Markox chain."""
        w, vr = np.linalg.eig(Pi.T)
        lambda_bar = vr[:,np.isclose(w,1.0)]
        lambda_bar /= sum(lambda_bar)
        lambda_bar = lambda_bar.flatten()
        return lambda_bar
    

    def initialize_b_vec(self):
        """Initialize a random starting state."""
        N = int(np.sum(100*S*lambda_bar))
        [np.exp(self.std*np.random.randn(S,round(b)))+self.mean
         for b in self.lambda_bar*self.N/self.S]
        
        
    def get_r_and_w(self):
        """Calculate r and w at the current state."""
        K = sum(list(self.state.sum()))
        L = (self.nvec*self.S).sum()
        r = self.alpha*self.A*(L/K)**(1-self.alpha)-self.delta
        w = (1-self.alpha)*self.A*(K/L)**self.alpha
        return r, w
            
            
            
# Define the Household parameters
N = 100
S = 3
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.6, 0.4],
               [0.4, 0.6]])
e_jt = np.array([0.8, 1.2])
mean = 0.0
std = 0.5

# S = 4
# J = 7
# Pi = np.array([[0.40,0.30,0.24,0.18,0.14,0.10,0.06],
#                [0.30,0.40,0.30,0.24,0.18,0.14,0.10],
#                [0.24,0.30,0.40,0.30,0.24,0.18,0.14],
#                [0.18,0.24,0.30,0.40,0.30,0.24,0.18],
#                [0.14,0.18,0.24,0.30,0.40,0.30,0.24],
#                [0.10,0.14,0.18,0.24,0.30,0.40,0.30],
#                [0.06,0.10,0.14,0.18,0.24,0.30,0.40]])
# Pi = (Pi.T/Pi.sum(1)).T
# e_jt = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5])

household_params = (N, S, J, beta_annual, sigma, Pi, e_jt, mean, std)

# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05

firm_params = (A, alpha, delta_annual)

# SS parameters

og = OG(household_params, firm_params)
og.SS()

og.K
og.TPI



