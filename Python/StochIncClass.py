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
        self.beta = beta_annual**(80/self.S)

        #Set labor exogenously
        nvec = np.ones(S)
        nvec[2*S/3:] = nvec[2*S/3:]*.3
        self.nvec = nvec

        self.lambda_bar = self.get_lambda_bar()

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params

        self.delta = 1-(1-delta_annual)**(80/self.S)

    

    def set_state(self):
        '''
        Here is some kind of function that sets all values, after
        parameters are all set
        '''
        self.initialize_b_vec()
        self.get_r_and_w()
        #self.get_c()


    def get_lambda_bar(self):
        """Compute the ergodic distribution of the Markox chain."""
        w, vr = np.linalg.eig(Pi)
        lambda_bar = vr[:,np.isclose(w,1.0)]
        lambda_bar /= sum(lambda_bar)
        lambda_bar = lambda_bar.flatten()
        return lambda_bar
    
    def initialize_b_vec(self):
        """Initialize a random starting state."""
        e0 = np.ones((self.S,self.J))*float(self.N)/(self.S*self.J)
        b0 = [list(np.random.rand(e)) for e in e0[1:].ravel()]
        b0 = [[0],[0]]+b0
        b0 = np.array(b0)
        self.b = b0

    def get_r_and_w(self, percent=False):
        """Calculate r and w at the current state."""
        if percent:
            self.K = sum(self.b)
        else:
            self.K = np.array(self.b.sum()).sum()
        self.L = sum(self.nvec)
        self.r = self.alpha*self.A*(self.L/self.K)**(1-self.alpha)-self.delta
        self.w = (1-self.alpha)*self.A*(self.K/self.L)**self.alpha
        
    def get_euler_errors(self, b):
        """Compute the euler errors."""
        
        
    def update_state(self):
        """Solve the individual lifetime problems."""
        self.b = fsolve(get_euler_errors)
        
    def SS(self, SS_params):
        """Find the steady state."""
        tol = SS_params
        diff = 1
        while diff<tol:
            current_state = self.state
            self.update_state()
            diff = np.linalg.norm(current_state-self.state)

    def _calc_u(self):
        """
        Calculates the utility, given consumption
        """
        utility = (self.c**(1-self.sigma)-1)/(1-self.sigma)
        return utility

    def get_c(self):
        '''
        Returns S length Consumption vector
        '''
        r, w = self.r, self.w
        b = np.copy(self.b)

        b_s = np.vstack(np.zeros((1,J), b))
        b_sp1 = np.append(b, np.zeros((1,J)))
        self.c = ((1 + r) * b_s + w * nvec - b_sp1)
        self.c_cstr = cvec <= 0

# Define the Household parameters
N = 800
S = 80
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.6, 0.4],
               [0.4, 0.6]])

# lambda_bar = get_lambda_bar(Pi)
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

# lambda_bar = get_lambda_bar(Pi)
# e_jt = np.array([0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5])

household_params = (N, S, J, beta_annual, sigma, Pi, e_jt)

# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05

firm_params = (A, alpha, delta_annual)


# SS parameters
b_guess = np.ones((S, J))*.0001
b_guess[0] = np.zeros(J)
SS_tol = 1e-10
rho = .5

#calculation
og = OG(household_params, firm_params)




