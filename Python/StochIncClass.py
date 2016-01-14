import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from scipy.stats import gamma
from quantecon import markov

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

        
    def calc_c(self,b, m):
        b0 = np.r_[0,b[:-1]]
        b1 = b
        e = self.abilities[:-1,m]
        r, w, nvec = self.r, self.w, self.nvec   
        c = (1+r)*b0 + nvec[:-1]*e*w-b1
        c_mask = c<0.0
        c[c_mask] = 0.0
        return c, c_mask

    
    def calc_Ec1(self, b, m):
        b0 = b
        b1 = np.r_[b[1:],0]
        pi = Pi[self.shocks[:-1,m]]
        r, w, nvec = self.r, self.w, self.nvec
        Ec1 = (pi*(nvec[1:,None]*self.e*w+(1+r)*b0[:,None]-b1[:,None])).sum(1)
        Ec1_mask = Ec1<0.0
        Ec1[Ec1_mask] = 0.0
        return Ec1, Ec1_mask


    def eul_err(self, b, m):
        beta, r = self.beta, self.r
        Ec1, Ec1_mask = self.calc_Ec1(b, m)
        c, c_mask = self.calc_c(b, m)
        eul_err = beta*(1+r)*(Ec1)**(-sigma) - c**(-sigma)
        eul_err[Ec1_mask] = 9999.
        eul_err[c_mask] = 9999.
        return eul_err


    def solve_1_agent(self,m):
        b0 = np.ones(self.S-1)*.1
#         if m!=0:
#             b0 = self.b_vec[1:,m-1]
        b, info, ier, mesg = fsolve(self.eul_err, b0, args=(m), full_output=1)
        return b, ier, mesg


    def update(self):
        """Update b_vec to the next period."""
        for m in range(5):
            self.b_vec[1:,m], ier, mesg = self.solve_1_agent(m)
#             print mesg
#             print self.b_vec[:,:3]
            if ier!=1:
                print "The fsolve didn't converge."

    def calc_ss(self, tol=1e-10, maxiter=100):
        self.set_state()
        self.r, self.w = 2., .2
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
#og.update()
#og.update()
#og.plot()
