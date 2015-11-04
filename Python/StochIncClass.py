import time
import numpy as np
from scipy.optimize import fsolve
from scipy.special import beta
import gb2library

class OG(object):
    '''
    Overlapping generations object
    Attributes:
        self.N - Int, Number of unique agents in model (100,000)
        self.S - Int, Number of years agent live (80)
        self.J - Int, Number of ability types
        self.beta_annual - Float, Annual discount rate
        self.sigma - 
        self.Pi - (SxJ) array, Markov probability matrix
        self.e_jt - 
        self.nvec
        self.lambda_bar - (J) array, ergodic distribution of SS ability
        self.alpha -
        self.delta_annual
        self.delta

    Methods:
        set_state:

    '''
    def __init__(self, household_params, firm_params):
        """Instatiate the state parameters of the OG model."""
        (self.N,
         self.S,
         self.J,
         self.beta_annual,
         self.sigma,
         self.Pi,
         self.e_jt,
         self.dist_params) = household_params
        self.beta = self.beta_annual**(80/self.S)

        nvec = np.ones(S)
        nvec[2*S/3:] = nvec[2*S/3:]*.3
        self.nvec = nvec
        self.lambda_bar = self.get_lambda_bar(self.Pi)

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params
        self.delta = 1-(1-self.delta_annual)**(80/self.S)
        self.set_state()
    

    def set_state(self):
        """Set initial state and r and w."""
        self.get_mean_GB2()
        self.initialize_b_vec()
        #self.get_r_and_w()
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
        '''
        self.b_vec = [np.exp(self.std*np.random.randn(S,round(b))+
            self.mean) for b in self.lambda_bar*self.N/self.S]
        '''
        self.b_vec = np.random.gamma(2,6,(self.S, self.J, 
            np.max(self.lambda_bar)*(self.N/self.S*self.J)))



    def get_mean_GB2(self):
        '''
        calculates the mean for a GB2 distribution
        '''
        a,b,p,q = self.dist_params
        self.mean = (b*beta(p+(1/p),q-(1/p)))/(beta(p,q))

        
    def get_r_and_w(self):
        """Calculate r and w at the current state."""
        self.K = sum(list(self.state.sum()))
        self.L = (self.nvec*self.S).sum()
        self.r = self.alpha*self.A*(L/K)**(1-self.alpha)-self.delta
        self.w = (1-self.alpha)*self.A*(K/L)**self.alpha


# Define the Household parameters
N = 200000
S = 80
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.1, 0.9],
               [0.6, 0.4]])
std = .5
mean = 0

dist_params_init = np.array([1, 500, .5, 1000])
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

household_params = (N, S, J, beta_annual, sigma, Pi, e_jt, dist_params_init)

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
print og.b_vec.shape
