import time
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import InterpolatedUnivariateSpline

class OG(object):
    '''
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
        #Make sure that N is divisible by S
        self.N = (self.N/self.S)*self.S

        nvec = np.ones(S)
        nvec[2*S/3:] = nvec[2*S/3:]*.3
        self.nvec = nvec
        self.lambda_bar = self.get_lambda_bar(self.Pi)

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params
        self.delta = 1-(1-self.delta_annual)**(80/self.S)
        self.initialize_b()
        self.set_state()
    

    def set_state(self):
        """Set initial state and r and w."""
        self.L = self.nvec.sum() * self.N
        self.K = self.b_vec.sum()
        self.get_r_and_w()


    def get_lambda_bar(self, Pi):
        """Compute the ergodic distribution of the Markox chain."""
        w, vr = np.linalg.eig(Pi.T)
        lambda_bar = vr[:,np.isclose(w,1.0)]
        lambda_bar /= sum(lambda_bar)
        lambda_bar = lambda_bar.flatten()
        return lambda_bar
    

    def initialize_b(self):
        """
        Initialize a random starting state.
        self.b_vec = (3,self.N) array where columns are:
            age, ability type, initial capitalstock
        """
        #TODO make this a panda
        b = np.random.gamma(2,6,self.N)
        skip = self.N/self.S
        ages = np.ones(self.N)
        for i in xrange(self.S):
            i += 1
            ages[i*skip:(i+1)*skip] = i+1
            test = ages[ages==i]
        self.abilities = np.ones(self.N)
        a_dist = np.random.multinomial(self.N, self.lambda_bar)
        for n in a_dist: 
            self.abilities[n:] += 1
        np.random.shuffle(self.abilities)
        self.b_vec = np.concatenate((ages, self.abilities, b), axis=1)
        self.b_vec = self.b_vec.reshape(3,self.N).T
        
    def get_r_and_w(self):
        """Calculate r and w at the current state."""
        A, alpha, delta, L, K = self.A, self.alpha, self.delta, self.L, self.K
        self.r = alpha * A * ((L/K)**(1-alpha)) - delta
        self.w = (1-alpha)*A*((K/L)**alpha)

    def calc_cs(self, b_s1, b_s, s, j):
        beta, r, w, nvec, e_j = self.beta, self.r, self.w, self.nvec, self.e_jt    
        c_s = (1+r)*b_s + nvec[s]*e_j[j]*w-b_s1
        cs_mask = c_s[c_s<0]
        c_s[cs_mask] = .00001
        return c_s, cs_mask

    def calc_cs1(self, b_s1, s, j, phi):
        beta, r, w, nvec, e_j = self.beta, self.r, self.w, self.nvec, self.e_jt
        nvec = np.concatenate((nvec,[0]))
        b_s1 = np.concatenate((b_s1,[0]))
        if phi==None:
            c_s1 = np.zeros_like(b_s1)
            for i in xrange(self.J):
                c_j = (1+r)*b_s1 + nvec[s+1]*e_j[i]*w
                cs1_mask = c_j<0
                c_j[cs1_mask] = .00001
                c_s1 += c_j
        else:
            c_s = np.zeros_like(b_s1)
            for i in xrange(self.J):
                c_j = (1+r)*b_s1 + nvec[s+1]*e_j[i]*w - phi(b_s1)
                cs1_mask = c_j<0
                c_j[cs1_mask] = .00001
                c_s1 += c_j
        return c_s1, cs1_mask

    def eul_err(self, b_s1, b_s, phi, s, j):
        beta, r = self.beta, self.r
        c_s1, cs1_mask = self.calc_cs1(b_s1, s, j, phi)
        c_s, cs_mask = self.calc_cs(b_s1, b_s, s, j)
        eul_err = beta*(1+r)*(c_s1)**(-sigma) - c_s**(-sigma)
        return eul_err

            
    def update(self):
        """Update b_vec to the next period."""
        #TODO use the panda
        self.set_state()
        b_vec_new = np.ones_like(self.b_vec)*999
        phi = [None]*self.J
        for s in xrange(self.S-1, 0, -1):
            s_mask = self.b_vec[:,0]==s
            for j in xrange(1,self.J+1):
                j_mask = self.b_vec[s_mask][:,1]==j
                phi_j = phi[j]
                b_s = self.b_vec[s_mask][j_mask][:,2]
                #TODO Make this draw from previous distribution for guess
                guess = b_s
                b_s1 = fsolve(self.eul_err, guess, args=(b_s, phi_j, s, j))
                phi[j] = InterpolatedUnivariateSpline(b_s, b_s1)
                self.b_vec[s_mask][j_mask][:,0] += 1
                self.b_vec[s_mask][j_mask][:,1] = np.multinomial(self.N/self.S/self.lambda_bar[j], self.Pi[j])
                self.b_vec[s_mask][j_mask][:,2] = b_s1
        S_mask = self.b_vec[:,0]==self.S
        self.b_vec[S_mask][:,0] = 0
        a_dist = np.multinomial(self.N/self.S, self.lambda_bar)
        for a in a_dist:
            self.b_vec[S_mask][:,1][a:] += 1.0
        self.b_vec[S_mask][:,2] = 0.0


# Define the Household parameters
N = 200000
S = 80
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.1, 0.9],
               [0.6, 0.4]])
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

#calculation
og = OG(household_params, firm_params)
