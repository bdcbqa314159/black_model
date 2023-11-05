# pylint: disable-all

import numpy as np
from scipy.stats import norm

def d_black_1(F_t, K, sigma, t, T):
    B = sigma*np.sqrt(T-t)
    A = np.log(F_t/K)/B + 0.5*B
    return A

def d_black_2(F_t, K, sigma, t, T):
    return d_black_1(F_t, K, sigma, t, T) - sigma*np.sqrt(T-t)

def discount(r,t,T):
    return np.exp(-r*(T-t))

def N(x):
    return norm.cdf(x)

def call(F_t, K, r, sigma, t, T):
    d_1 = d_black_1(F_t, K, sigma, t, T)
    d_2 = d_black_2(F_t, K, sigma, t, T)
    return discount(r,t,T)*(F_t*N(d_1) - K*N(d_2))

def put(F_t, K, r, sigma, t, T):
    d_1 = d_black_1(F_t, K, sigma, t, T)
    d_2 = d_black_2(F_t, K, sigma, t, T)
    return discount(r,t,T)*(-F_t*N(-d_1) + K*N(-d_2))

if __name__ == '__main__':
    print('black model')