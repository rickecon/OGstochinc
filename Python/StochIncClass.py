import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import gamma
from scipy.stats import gengamma


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
         self.e_jt) = household_params
        self.beta = self.beta_annual**(80/self.S)
        # Force N to be divisible by S
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
        

    def get_lambda_bar(self, Pi):
        """Compute the ergodic distribution of the Markox chain."""
        w, vr = np.linalg.eig(Pi.T)
        lambda_bar = vr[:,np.isclose(w,1.0)]
        lambda_bar /= sum(lambda_bar)
        lambda_bar = lambda_bar.flatten()
        return lambda_bar
    
    
    def set_state(self):
        """Set initial state and r and w."""
        self.L = self.nvec.sum() * self.N
        self.K = self.b_vec.sum()
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
        # TODO make this a panda
        intial_params = [3.,.001,.5]
        a, mean, b = intial_params
        b = gamma.rvs(a ,loc=mean, scale=b, size=self.N)
        skip = self.N/self.S
        b[:skip] = 0
        ages = np.zeros(self.N)
        for i in xrange(1,self.S):
            ages[i*skip:(i+1)*skip] = i
        self.abilities = np.ones(self.N)
        a_dist = np.random.multinomial(self.N, self.lambda_bar)
        for n in np.cumsum(a_dist[:-1]): 
            self.abilities[n:] += 1
        np.random.shuffle(self.abilities)
        self.b_vec = np.concatenate((ages, self.abilities, b), axis=1)
        self.b_vec = self.b_vec.reshape(3,self.N).T
        #Initilize the parameters vectors
        self.gamma_par = np.zeros((self.S, self.J, 3))
        self.gen_gamma_par = np.ones((self.S, self.J, 4))
        for s in xrange(self.S):
            for j in xrange(self.J):
                self.gamma_par[s,j,:] = intial_params
                self.gen_gamma_par[s,j,:-1] = intial_params

        
    def calc_cs(self, b_s1, b_s, s, j):
        r, w, nvec, e_j = self.r, self.w, self.nvec, self.e_jt    
        c_s = (1+r)*b_s + nvec[s]*e_j[j-1]*w-b_s1
        cs_mask = c_s<0.0
        c_s[cs_mask] = 0.0
        return c_s, cs_mask

    
    def calc_Ecs1(self, b_s1, s, j, phi):
        r, w, nvec, e_j = self.r, self.w, self.nvec, self.e_jt
        if phi==None:
            Ec_s1 = np.zeros_like(b_s1)
            for i in xrange(self.J):
                c_j = (1+r)*b_s1 + nvec[s+1]*e_j[i]*w
                Ec_s1 += c_j*Pi[j-1,i]
        else:
            Ec_s1 = np.zeros_like(b_s1)
            for i in xrange(self.J):
                c_j = (1+r)*b_s1 + nvec[s+1]*e_j[i]*w - phi(b_s1)
                Ec_s1 += c_j*Pi[j-1,i]
        Ecs1_mask = Ec_s1<0.0
        Ec_s1[Ecs1_mask] = 0.0
        return Ec_s1, Ecs1_mask

    
    def eul_err(self, b_s1, b_s, phi, s, j):
        beta, r = self.beta, self.r
        Ec_s1, Ecs1_mask = self.calc_Ecs1(b_s1, s, j, phi)
        c_s, cs_mask = self.calc_cs(b_s1, b_s, s, j)
        eul_err = beta*(1+r)*(Ec_s1)**(-sigma) - c_s**(-sigma)
        eul_err[Ecs1_mask] = 9999.
        eul_err[cs_mask] = 9999.
        return eul_err

            
    def update(self):
        """Update b_vec to the next period."""
        #TODO use the panda
        self.set_state()
        phi = [None]*self.J
        print phi
        for s in xrange(self.S-2, -1, -1):
            for j in xrange(1,self.J+1):
                print s, j
                # Create the masks that will be used for this (s,j) combination.
                mask = (self.b_vec[:,1]==j) & (self.b_vec[:,0]==s)
                num = np.sum(mask)
                b_mask = np.zeros((self.N,3),dtype='bool')
                b_mask[:,2] = mask
                j_mask = np.zeros((self.N,3),dtype='bool')
                j_mask[:,1] = mask
                # Solve the consumption problem.
                phi_j = phi[j-1]
                b_s = self.b_vec[b_mask]
                s_ind = b_s.argsort()
                # TODO Make this draw from previous distribution for guess.
                g_params = self.gamma_par[s,j-1,:]
                a, mean, b = g_params
                print mean
                guess = np.ones(num)*mean
                b_s1 = fsolve(self.eul_err, guess, args=(b_s, phi_j, s, j))
                print fsolve(self.eul_err, guess, args=(b_s, phi_j, s, j), full_output = 1)[-1]
                # Create the policy function.
                # Is Policy function working for age = 0/1?
                phi[j-1] = UnivariateSpline(b_s[s_ind], b_s1[s_ind])
                #plt.plot(b_s,phi[j-1](b_s))
                #plt.title("b_s interpolated")
                # plt.show()
                # Update the (s,j) part of the b_vec.
                a_dist = np.random.multinomial(num, self.Pi[j-1])
                new_j = np.ones(num)
                for a in np.cumsum(a_dist[:-1]):
                    new_j[a:] += 1.0
                self.b_vec[j_mask] = np.random.permutation(new_j)
                self.b_vec[b_mask] = b_s1[s_ind]
        mask = self.b_vec[:,0]==self.S-1
        b_mask = np.zeros((self.N,3),dtype='bool')
        b_mask[:,2] = mask
        j_mask = np.zeros((self.N,3),dtype='bool')
        j_mask[:,1] = mask
        num = np.sum(mask)
        # Update ages.
        self.b_vec[:,0] += 1.0
        self.b_vec[:,0] %= self.S
        # Update b and j for the dead.
        a_dist = np.random.multinomial(self.N/self.S, self.lambda_bar)
        new_j = np.ones(num)
        for a in np.cumsum(a_dist[:-1]):
            new_j[a:] += 1.0
        self.b_vec[j_mask] = np.random.permutation(new_j)
        self.b_vec[b_mask] = 0.0
        self.phi = phi
        self.fit_params()

    def gamma_fit(self, b_vec, params, s, j):
        a, mean, scaler = params
        params = gamma.fit(b_vec, a, loc=mean, scale=scaler)
        print params
        self.gamma_par[s,j-1,:] = params

    def gen_gamma_fit(self, b_vec, params, s, j):
        a, mean, scaler, c = params
        params = gengamma.fit(b_vec, a, c, loc=mean, scale=scaler)
        self.gen_gamma_par[s,j-1,:] = params
        
    def fit_params(self):
        '''
        Fits each s,j group to a gamma distribution
        '''
        print 'Fitting curves'
        for s in xrange(1,self.S):
            for j in xrange(1,self.J+1):
                g_params = self.gamma_par[s,j-1,:]
                gg_params = self.gen_gamma_par[s,j-1,:]
                mask = (self.b_vec[:,1]==j) & (self.b_vec[:,0]==s)
                num = np.sum(mask)
                b_mask = np.zeros((self.N,3),dtype='bool')
                b_mask[:,2] = mask
                j_mask = np.zeros((self.N,3),dtype='bool')
                j_mask[:,1] = mask
                b_vec = self.b_vec[b_mask]
                self.gamma_fit(b_vec, g_params, s, j)
                self.gen_gamma_fit(b_vec, gg_params, s, j)
        print 'curves fit'
        pass

    def plot(self):
        for s in xrange(1,self.S):
            for j in xrange(1,self.J+1):
                print s
                mask = (self.b_vec[:,1]==j) & (self.b_vec[:,0]==s)
                num = np.sum(mask)
                b_mask = np.zeros((self.N,3),dtype='bool')
                b_mask[:,2] = mask
                j_mask = np.zeros((self.N,3),dtype='bool')
                j_mask[:,1] = mask
                b_vec = self.b_vec[b_mask]
                print b_vec.shape
                if len(b_vec) != 0:
                    x_plot = np.linspace(.1,10,1000)
                    g_params = self.gamma_par[s,j-1,:]
                    a, mean, b = g_params
                    g_x = gamma.pdf(x_plot,a,loc=mean,scale=b)
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    ax.hist(b_vec, bins=50, normed=True, label="data")
                    ax.plot(x_plot, g_x, 'r-', label="pdf")
                    ax.legend(loc='best')
                    plt.show()
        pass

# Define the Household parameters
N = 10000
S = 10
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.4, 0.6],
               [0.6, 0.4]])
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
b_guess = np.ones((S, J))*.0001
b_guess[0] = np.zeros(J)
SS_tol = 1e-10
rho = .5

#calculation
og = OG(household_params, firm_params)
# og.update()
#og.update()
#og.plot()

