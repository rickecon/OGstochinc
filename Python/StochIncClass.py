import time
import numpy as np
from scipy.optimize import fsolve
import gb2library
from scipy.interpolate import InterpolatedUnivariateSpline

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
        """Instantiate the state parameters of the OG model."""
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
        self.lambda_bar = self.get_lambda_bar(self.Pi)

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params
        self.delta = 1-(1-self.delta_annual)**(80/self.S)
        self.set_state()
    

    def set_state(self):
        """Set initial state and r and w."""
        self.initialize_bvec()
        self.get_r_and_w()


    def get_lambda_bar(self, Pi):
        """Compute the ergodic distribution of the Markox chain."""
        w, vr = np.linalg.eig(Pi.T)
        lambda_bar = vr[:,np.isclose(w,1.0)]
        lambda_bar /= sum(lambda_bar)
        lambda_bar = lambda_bar.flatten()
        return lambda_bar
    

    def initialize_bvec(self):
        """Initialize a random starting state."""
        self.bvec = np.random.gamma(2,6,(self.S, self.J, 
            np.max(self.lambda_bar)*(self.N/self.S)+10))


    def get_r_and_w(self):
        """Calculate r and w at the current state."""
        K = self.bvec.sum()
        L = self.N/self.S*self.nvec.sum()
        r = self.alpha*self.A*(L/K)**(1-self.alpha)-self.delta
        w = (1-self.alpha)*self.A*(K/L)**self.alpha
        return r, w
    
    
    def update(self):
        """Update bvec to the next period."""
        bvec_new = np.ones_like(self.bvec)*999
        phi = [None]*self.J
        for s in xrange(self.S-1, 0, -1):
            for j in xrange(self.J):
                phi_j = phi[j]
                b_s = self.bvec[s,j]
                mask = b_s<999
                b_s = b_s[mask]
                guess = b_s/2.
                b_s1 = fsolve(eul_err, guess, args=(b_s, phi_j, j))
                phi[j] = InterpolatedUnivariateSpline(b_s, b_s1)
                bvec_new[s+1,j][mask] = b_s1
        self.bvec = self.randomize(bvec_new)


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