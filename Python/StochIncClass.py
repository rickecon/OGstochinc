import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from scipy.stats import gamma
from quantecon import markov
from scipy.interpolate import UnivariateSpline

class OG(object):
    """
    """
    def __init__(self, household_params, firm_params):
        """Instantiate the state parameters of the OG model."""
        (self.N,
         self.S,
         self.J,
         self.beta_annual,
         self.sigma,
         self.Pi,
         self.e) = household_params
        
        self.beta = self.beta_annual**(80/self.S)

        nvec = np.ones(S)
        nvec[2*S/3:] = nvec[2*S/3:]*.3
        self.nvec = nvec
        
        MC = markov.core.MarkovChain(self.Pi)
        self.lambda_bar = MC.stationary_distributions
        weights = (self.lambda_bar/(self.lambda_bar.min())*self.N).astype('int')
        self.M = np.sum(weights)
        initial_e = np.zeros(self.M)
        for weight in np.cumsum(weights[0][1:]):
            initial_e[weight:] += 1
        self.shocks = MC.simulate(self.S, initial_e).T
        self.abilities = self.e[self.shocks]

        (self.A,
         self.alpha,
         self.delta_annual) = firm_params
        
        self.delta = 1-(1-self.delta_annual)**(80/self.S)
        
        self.initialize_b()
        self.set_state()
    
    
    def set_state(self):
        """Set initial state and r and w."""
        self.L = self.nvec.sum() * self.N
        weightedK = np.array([self.b_vec[self.shocks==i].sum() for i in range(self.J)])
        self.K = np.sum(self.lambda_bar*weightedK)/self.S
        self.get_r_and_w()
        
        
    def get_r_and_w(self):
        """Calculate r and w at the current state."""
        A, alpha, delta, L, K = self.A, self.alpha, self.delta, self.L, self.K
        self.r = alpha * A * ((L/K)**(1-alpha)) - delta
        self.w = (1-alpha)*A*((K/L)**alpha)

    def initialize_b(self):
        """
        Initialize a random starting state.
        self.b_vec = (3,self.N) array where columns are:
            age, ability type, initial capitalstock
        """
#         intial_params = [3.,.001,.5]
#         a, mean, scale = intial_params
#         self.b_vec = gamma.rvs(a , loc=mean, scale=scale, size=(self.S,self.M))
        self.b_vec = np.ones((self.S, self.M))*.1
        self.b_vec[0] = 0.0

        
    def calc_c(self, b1, s, j, psi, b0):
        e, r, w, nvec = self.e[j], self.r, self.w, self.nvec 
        c = nvec[s]*e*w + (1+r)*b0 - b1
        c_mask = c<0.0
        c[c_mask] = 0.0
        return c**(-sigma), c_mask

    
    def calc_Ec1(self, b1, s, j, psi):
        r, w, nvec = self.r, self.w, self.nvec
        if psi==None:
            b2 = np.zeros(101)
        else:
            b2 = psi(b1)
        c1 = (nvec[s-1]*self.e*w+(1+r)*b1[:,None]-b2[:,None])
        Ec1_mask = ((c1<0.0).sum(1)).astype("bool")
        c1[Ec1_mask] = 0.0
        c1 = c1**(-sigma)
        Ec1 = (Pi[j]*c1).sum(1)
        return Ec1, Ec1_mask


    def eul_err(self, b1, s, j, psi, grid):
        beta, r = self.beta, self.r
        Ec1, Ec1_mask = self.calc_Ec1(b1, s, j, psi)
        c, c_mask = self.calc_c(b1, s, j, psi, grid)
        eul_err = beta*(1+r)*(Ec1)-c
        eul_err[Ec1_mask] = 9999.
        eul_err[c_mask] = 9999.
        return eul_err

    def update_polfun(self):
        r, w, nvec, S = self.r, self.w, self.nvec, self.S
#         e_max = np.max(self.e)
#         bound = np.sum([(1+r)**i*w*nvec[S-1-i]*e_max for i in range(S)])
#         print bound
        bound = 2.
        grid = np.linspace(0, bound, 101)
        self.Psi = np.empty((self.S-1, self.J, 101))
        psi = None
        for j in range(self.J):
            for s in range(self.S-2, -1, -1):
                print s, j
                guess = np.ones(101)*.001
                self.Psi[s,j], info, ier, mesg = fsolve(self.eul_err, guess, args=(s,j,psi,grid), full_output=1) 
                if ier!=1:
                    print 'warning'
                psi = UnivariateSpline(self.Psi[s,j], grid)
    
    
    def update_bvec(self):
        for s in range(1, self.S):
            for j in range(self.J):
                psi = UnivariateSpline(Psi[s, j])
                mask = self.shocks[s]==j
                self.b_vec[mask] = psi(self.b_vec[mask])
        self.b_vec = np.roll(self.b_vec, axis=0)
        

    def calc_ss(self, tol=1e-10, maxiter=100):
        self.set_state()
        self.r, self.w = 1., .2
        diff = 1
        count = 0
        while diff>tol and count<maxiter:
            r0, w0 = self.r, self.w
            self.update()
            self.set_state()
            diff = max(self.r-r0, self.w-w0)
            count += 1
            print diff, count


# Define the Household parameters
N = 100
S = 10
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.4, 0.6],
               [0.6, 0.4]])
e_jt = np.array([0.8, 1.2])

# S = 10
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
rho = .5

#calculation
og = OG(household_params, firm_params)

