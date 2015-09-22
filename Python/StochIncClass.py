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
         self.e_jt) = household_params
        self.beta = self.beta_annual**(80/self.S)
        nvec = np.ones(S)
        nvec[2*S/3:] = nvec[2*S/3:]*.3
        self.nvec = nvec
        self.lambda_bar = self.get_lambda_bar(Pi)

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params
        self.delta = 1-(1-self.delta_annual)**(80/self.S)
    
    def get_lambda_bar(self, Pi):
        """Compute the ergodic distribution of the Markox chain."""
        w, vr = np.linalg.eig(Pi)
        lambda_bar = vr[:,np.isclose(w,1.0)]
        lambda_bar /= sum(lambda_bar)
        lambda_bar = lambda_bar.flatten()
        return lambda_bar
    
    def initialize_state(self):
        """Initialize a random starting state."""
        S0 = np.random.multinomial(self.N, np.ones(self.S)/float(self.S))
        e0 = np.array([np.random.multinomial(s, self.lambda_bar) for s in S0])
        b0 = np.zeros_like(e0, dtype="object")
        b0[0] = np.array([tuple([0]*e) for e in e0[0]])
        b0[1:] = np.array([tuple(np.random.rand(e)*15) for e in e0[1:].ravel()]).reshape(S-1,J)
        self.state = b0
        self.S = S0
        
    def get_r_and_w(self, percent=False):
        """Calculate r and w at the current state."""
        if percent:
            K = sum(list(self.state.sum()))/self.N
        else:
            K = sum(list(self.state.sum()))
        L = (self.nvec*self.S).sum()
        r = self.alpha*self.A*(L/K)**(1-self.alpha)-self.delta
        w = (1-self.alpha)*self.A*(K/L)**self.alpha
        return r, w
        
    def get_euler_errors(self, b):
        """Compute the euler errors."""
        
        
#     def update_state(self):
#         """Solve the individual lifetime problems."""
#         for row in self.state:
#             for column in row:
                
#             # For s=S-1
#             c0 = w*e_jt+(1+r)*
#             new_state.append()
#             # For s=S don't do anything
#             new_state.append()
#         b = fsolve(get_euler_errors)
        
    def SS(self, SS_params):
        """Find the steady state."""
        tol = SS_params
        diff = 1
        while diff<tol:
            current_state = self.state
            self.update_state()
            diff = np.linalg.norm(current_state-self.state)
            
            
            
# Define the Household parameters
N = 100
S = 3
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.6, 0.4],
               [0.4, 0.6]])
e_jt = np.array([0.8, 1.2])

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

household_params = (N, S, J, beta_annual, sigma, Pi, e_jt)

# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05

firm_params = (A, alpha, delta_annual)

# SS parameters