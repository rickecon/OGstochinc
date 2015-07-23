'''
------------------------------------------------------------------------
Last updated 7/8/2015

All the functions for the SS and TPI computation from Section 2.
------------------------------------------------------------------------
'''
# Import Packages
import numpy as np
import scipy.optimize as opt
import Sec2ssfuncs as s2ssf
reload(s2ssf)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def get_cbepath(ints, params, rpath, wpath, Gamma1):
    S, T = ints
    beta, sigma, TPI_tol = params
    cpath = np.zeros((S, T))
    bpath = np.zeros((S-1, T-1))
    EulErrPath = np.zeros((S-1, T-1))
    # Solve the incomplete remaining lifetime decisions
    cpath[S-1,0] = (1 + rpath[0]) * Gamma1[S-2] + wpath[0]
    pl_ints = np.array([S])
    pl_params = np.array([beta, sigma, TPI_tol])
    for p in xrange(2, S):
        bveclf, cveclf, b_err_veclf = paths_life(pl_ints, pl_params, S-p+1, Gamma1[S-p-1], rpath[:p-1], wpath[:p-1])
#         # Insert the vector lifetime solutions diagonally (twist donut)
#         # into the cpath, bpath, and EulErrPath matrices
#         DiagMaskb = np.eye(p-1, dtype=bool)
#         DiagMaskc = np.eye(p, dtype=bool)
#         DiagMaskerr = np.eye(p-1, dtype=bool)
#         bpath[S-p:, 1:p-1] = Diagmaskb * bveclf + bpath[S-p:, 1:p-1]
#         cpath[S-p:, :p-1] = Diagmaskc * bveclf + cpath[S-p:, :p-1]
#         EulErrPath[S-p:, 1:p-1] = Diagmaskerr * b_err_veclf + EulErrPath[S-p:, 1:p-1]
#     # Solve for complete lifetime decisions of agents born in periods
#     # 1 to T and insert the vector lifetime solutions diagonally (twist
#     # donut) into the cpath, bpath, and EulErrPath matrices
#     DiagMaskb = np.eye(S-1, dtype=bool)
#     DiagMaskc = np.eye(S, dtype=bool)
#     DiagMaskerr = np.eye(S-1, dtype=bool)
#     for t in xrange(1, T+1):
#         bveclf, cveclf, b_err_veclf = paths_life(pl_ints, pl_params, 1, 0,
#             rpath[t-1:t+S-1], wpath[t-1:t+S-1])
#         bpath[:, t:t+S-1] = Diagmaskb * bveclf + bpath[:, t:t+S-1]
#         cpath[:, t-1:t+S-1] = Diagmaskc * bveclf + cpath[:, t-1:t+S-1]
#         EulErrPath[:, t:t+S-1] = Diagmaskerr * b_err_veclf + EulErrPath[:, t:t+S-1]
    return cpath, bpath, EulErrPath


# def paths_life(ints, params, beg_age, beg_wealth, rpath, wpath):
#     '''
#     Solve for the remaining lifetime savings decisions of an individual
#     who enters the model at age beg_age, with corresponding initial
#     wealth beg_wealth.

#     Inputs:
#         params     = [4,] vector, parameters [S, beta, sigma, TPI_tol]
#         beg_age    = integer in [1,S-1], beginning age of remaining life
#         beg_wealth = scalar, beginning wealth at beginning age
#         rpath      = [S-beg_age+1,] vector, remaining lifetime interest
#                      rates
#         wpath      = [S-beg_age+1,] vector, remaining lifetime wages

#     Functions called:
#         get_rpath
#         get_wpath
#         [TODO]

#     Objects in function:
#         p = integer >= 2, remaining periods in life

#     Returns: bpath, cpath, b_err_vec
#     '''
#     S = ints
#     beta, sigma, TPI_tol = params
#     if beg_age == 1 and beg_wealth != 0:
#         sys.exit("Beginning wealth is nonzero for age s=1.")
#     if len(rpath) != S-beg_age+1:
#         sys.exit("Beginning age and length of rpath do not match.")
#     if len(wpath) != S-beg_age+1:
#         sys.exit("Beginning age and length of wpath do not match.")
#     p = int(S - beg_age + 1)
#     b_guess = 0.01 * np.ones(p - 1)
#     eullf_ints = np.array([p])
#     eullf_params = np.array([beta, sigma])
#     bpath = opt.fsolve(LfEulerSys, b_guess, args=(eullf_ints, eullf_params, beg_wealth, rpath, wpath), xtol=TPI_tol)
#     cpath, c_constr = get_cvec_lf(p, rpath, wpath,
#                                   np.append(beg_wealth, bpath))
#     b_err_ints = np.array([p])
#     b_err_params = np.array([beta, sigma])
#     b_err_vec = get_b_errors(b_err_ints, b_err_params, rpath[1:], cvec, c_constr,
#                              diff=True)
#     return bpath, cpath, b_err_vec


# def get_cvec_lf(p, rpath, wpath, bvec):
#     '''
#     Generates vector of remaining lifetime consumptions from individual
#     savings, and the time path of interest rates and the real wages

#     Inputs:
#         p     = integer in [2,80], number of periods remaining in
#                 individual life
#         rpath = [p,] vector, remaining interest rates
#         wpath = [p,] vector, remaining wages
#         bvec =  [p,] vector, remaining savings including initial savings

#     Functions called: None

#     Objects in function:
#         c_constr = [p,] boolean vector, =True if element c_s <= 0
#         b_s      = [p,] vector, bvec
#         b_sp1    = [p,] vector, last p-1 elements of bvec and 0 in last
#                    element
#         cvec     = [p,] vector, remaining consumption by age c_s

#     Returns: cvec, c_constr
#     '''
#     c_constr = np.zeros(p, dtype=bool)
#     b_s = bvec
#     b_sp1 = np.append(bvec[1:], [0])
#     cvec = (1 + rpath) * b_s + wpath - b_sp1
#     if cvec.min() <= 0:
#         print 'initial guesses and/or parameters created c<=0 for some agent(s)'
#         c_constr = cvec <= 0
#     return cvec, c_constr



# def LfEulerSys(bvec, ints, params, beg_wealth, rpath, wpath):
#     '''
#     Generates vector of all Euler errors that characterize all
#     optimal lifetime decisions

#     Inputs:
#         bvec       = [p-1,] vector, remaining lifetime savings levels
#                      where p is the number of remaining periods
#         ints       = [1,] vector, parameters [S]
#         params     = [2,] vector, parameters [S, beta, sigma]
#         p          = integer in [2,80], remaining periods in life
#         beta       = scalar in [0,1), discount factor
#         sigma      = scalar > 0, coefficient of relative risk aversion
#         beg_wealth = scalar, wealth at the beginning of first age
#         rpath      = [y,] vector, interest rates over remaining life
#         wpath      = [y,] vector, wages rates over remaining life

#     Functions called:
#         ?

#     Objects in function:
#         ?

#     Returns: b_errors
#     '''
#     p = ints
#     beta, sigma = params
#     bvec2 = np.append(beg_wealth, bvec)
#     cvec, c_constr = get_cvec_lf(p, rpath, wpath, bvec2)
#     b_err_ints = np.array([p])
#     b_err_params = np.array([beta, sigma])
#     b_err_vec = get_b_errors(b_err_ints, b_err_params, rpath[1:], cvec, c_constr,
#                              diff=True)
#     return b_err_vec


def TPI(ints, params, Kpath_init, Gamma1, Lpath, graphs):
    '''
    Generates steady-state time path for all endogenous objects from
    initial state (K1, Gamma1) to the steady state.

    Inputs:
        ints       = [2,] vector, integer parameters [S, T]
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        params     = [11,] vector, parameters [beta, sigma, A, alpha,
                     delta, K1, K_ss, maxiter_TPI, mindist_TPI, xi,
                     TPI_tol]
        ?
        Kpath_init = [T+S-1,] vector, initial guess for the time path of
                     the aggregate capital stock
        [TODO]

    Functions called:
        s2ssf.get_r
        s2ssf.get_w
        get_cbepath

    Objects in function:
        cpath      = [S, T] matrix, equilibrium time path values of
                     individual consumption c_{s,t}
        bpath_new  = [S-1, T-1] matrix, equilibrium time path values of
                     individual savings b_{s+1,t+1}
        EulErrPath = [S-1, T-1] matrix, equilibrium time path values of
                     Euler errors corresponding to individual savings
                     b_{s+1,t+1}
        bpath      = [S-1, T] matrix, equilibrium time path values of
                     individual savings b_{s+1,t+1} including initial
                     distribution of savings Gamma1
        cpath      = [S, T] matrix, equilibrium time path values of
                     individual consumption c_{s,t}
        Kpath      = [T+S-1,] vector, equilibrium time path of the
                     aggregate capital stock
        rpath      = [T+S-1,] vector, equilibrium time path of the
                     interest rate
        wpath      = [T+S-1,] vector, equilibrium time path of the real
                     wage


    Returns: bpath, cpath, wpath, rpath, Kpath, EulErrpath
    '''
    S, T = ints
    beta, sigma, A, alpha, delta, K1, K_ss, maxiter_TPI, mindist_TPI, \
        xi, TPI_tol = params
    iter_TPI = int(0)
    dist_TPI = 10.
    Kpath_new = Kpath_init
    rp_params = np.array([A, alpha, delta])
    wp_params = np.array([A, alpha])
    cbe_ints = np.array([S, T])
    cbe_params = np.array([beta, sigma, TPI_tol])

    while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        rpath = s2ssf.get_r(rp_params, Kpath_init, Lpath)
        wpath = s2ssf.get_w(wp_params, Kpath_init, Lpath)
        cpath, bpath_new, EulErrPath = get_cbepath(cbe_ints, cbe_params,
                                       rpath, wpath, Gamma1)
        Kpath_new = np.append(K1, bpath_new.sum(axis=0).reshape(T-1),
                              K_ss * np.ones(S-1))
        # Check the distance of Kpath_new1
        dist_TPI = np.absolute((Kpath_new[1:T] - Kpath_init[1:T]) /
                   Kpath_init[1:T]).max()
        print 'iter: ', iter_TPI, ', dist: ', dist_TPI

    if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
        print 'TPI reached maxiter and did not converge.'
    if iter_TPI == maxiter_TPI and dist_TPI <= mindist_TPI:
        print 'TPI converged in the last iteration. Should probably increase maxiter_TPI.'
    Kpath = Kpath_new
    bpath = np.append(Gamma1.reshape((S-1, 1)), bpath_new, axis=1)

    return bpath, cpath, Kpath, rpath, wpath, EulErrPath
