import pandas as pd 
import numpy as np 
import gb2library as carla
import pandas.io.data as web 
from glob import glob
from pandas import Series, DataFrame, Index 
from scipy.special import gamma
import datetime
import matplotlib.pyplot as plt
from itertools import product
from pandas.io.data import DataReader
from scipy.special import betaln, gammaln, beta, gamma, betainc, gammainc, erf
from scipy import optimize as opt 


# Problem 44
incomes =  pd.io.parsers.read_table('./usincmoms.txt', names=["percent","incbins"])
# stats=incomes.describe()
# print stats

bins = (list(incomes['incbins']*.001)+[400])
widths=np.diff(bins)
widths[-3],widths[-2]=5,50
percent = np.array(incomes['percent'])
percent[-1], percent[-2]= percent[-1]/20., percent[-2]/10.
plt.bar(bins[:-1], percent, widths,align='center')
plt.title('Income Distribution')
# plt.show()

#Problem 45
x=np.array(incomes['incbins'])
moms=np.array(incomes['percent'])
w=np.identity(42)*moms
mu=np.log(69677)
sigma=.2*mu
theta0=[mu,sigma]

def ms(theta0):
    mu,sigma=theta0[0],theta0[1]
    mgen=carla.Ln(mu,sigma)
    mhat=mgen.pdf(x)*5000
    return mhat
# mhat=ms(theta0)
# print mhat

def e(theta0):
    m=(ms(theta0)-moms)/moms
    return m

def values(theta0):
    val=((e(theta0).T).dot(w)).dot(e(theta0))
    return val

theta=opt.minimize(values, theta0, method = 'Powell', tol = 1e-10, options = ({'maxiter': 50000, 'maxfev' : 50000}))
par=theta.x
mu1, sigma1= par[0],par[1]
print mu1,sigma1

bins=np.array(incomes['incbins']).flatten()
bins2=bins*.001
mgen1=carla.Ln(mu1,sigma1)
y1=mgen1.pdf(x)*5000
plt.plot(bins2,y1,"y")
bins = (list(incomes['incbins']*.001)+[400])
widths=np.diff(bins)
widths[-3],widths[-2]=5,50
percent = np.array(incomes['percent'])
percent[-1], percent[-2]= percent[-1]/20., percent[-2]/10.
plt.bar(bins[:-1], percent, widths,align='center')
plt.title('Income Distribution')
plt.show()

# Problem 46

alpha0=3
beta0=20000
init=[beta0,alpha0]

def GA(init): 
    GAgen=carla.Ga(beta0,alpha0)
    mhat=GAgen.pdf(x)*5000
    mhat[-1]=mhat[-1]*20
    mhat[-2]=mhat[-2]*10
    return mhat
print GA(init)

yeyeah=opt.minimize(values, [20000,3], method = 'Powell', tol = 1e-5, options = ({'maxiter': 50000, 'maxfev' : 50000}))
par=yeyeah.x
beta1, alpha1= par[0],par[1]
print beta1, alpha1

bins=np.array(incomes['incbins']).flatten()
bins2=bins*.001
whatwhat=carla.Ga(beta1,alpha1)
y2=whatwhat.pdf(x)*5000
plt.plot(bins2,y2,"y")
bins = (list(incomes['incbins']*.001)+[400])
widths=np.diff(bins)
widths[-3],widths[-2]=5,50
percent = np.array(incomes['percent'])
percent[-1], percent[-2]= percent[-1]/20., percent[-2]/10.
plt.bar(bins[:-1], percent, widths,align='center')
plt.title('Income Distribution')
plt.show()

#Problem 47


bins=np.array(incomes['incbins']).flatten()
bins2=bins*.001
whatwhat=carla.Ga(beta1,alpha1)
y2=whatwhat.pdf(x)*5000
plt.plot(bins2,y2,"y")

bins=np.array(incomes['incbins']).flatten()
bins2=bins*.001
mgen1=carla.Ln(mu1,sigma1)
y1=mgen1.pdf(x)*5000
plt.plot(bins2,y1,"y")
bins = (list(incomes['incbins']*.001)+[400])
widths=np.diff(bins)
widths[-3],widths[-2]=5,50
percent = np.array(incomes['percent'])
percent[-1], percent[-2]= percent[-1]/20., percent[-2]/10.
plt.bar(bins[:-1], percent, widths,align='center')
plt.title('Income Distribution')
plt.show()

