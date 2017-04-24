"""This file contains methods for finding deltas 
"""

import numpy as np
from numba import jit, float64, int16


@jit(float64[:](float64[:], float64[:], float64[:,:], float64[:,:], int16))
def simulate_shares(delta, prices, chars, P0, NS):
    """Takes guess for delta, ideosyncratic taste shocks
        prices, product characteristics, and number of simulated individuals
    return market shares implied by the model
    """
    #Num Products in this market
    J = len(prices)
    totals = np.zeros(J)
    D      = np.zeros(J)

    for i in range(NS):
        for j in range(J):
            D[j] = (delta[j] 
                    + np.dot(chars[j], P0[i][1:]) 
                    - prices[j]*P0[i][0])
        expD   = np.exp(D)
        condC  = np.divide(expD, (1+np.sum(expD)))
        totals = np.add(totals, condC)
    
    return np.divide(totals, NS)

@jit(float64[:,:](float64[:], float64[:], float64[:], float64[:,:], float64[:,:], int16))
def simulate_derivs(delta, beta, prices, chars, P0, NS):
    J = len(prices)
    
    derivs = np.zeros((J,J))
    F      = np.zeros(J)
        
    for i in range(NS):
        Dmu = beta[-1]-P0[i][0]
        for j in range(J):
            F[j] = (delta[j]
                    + np.dot(chars[j], P0[i][1:])
                    - prices[j]*P0[i][0])
        F = np.exp(F)
        F = np.divide(F, (1+np.sum(F)))
        
        for j in range(J):
            for q in range(j+1):
                if j != q:
                    derivs[j][q] += -F[j]*F[q]*Dmu
                else:
                    derivs[j][q] += F[j]*(1-F[j])*Dmu
    for j in range(J):
        for q in range(j):
            derivs[q][j] = derivs[j][q]

    return np.divide(derivs, NS)

@jit(float64[:,:](float64[:], float64[:], float64[:,:], float64[:], float64[:,:], int16[:,:], int16))
def make_Delta(delta, prices, chars, beta, P0, ownership, NS):
    """Returns the Delta matrix, where:
        Delta_jr = -ds_r/dp_j if r,j are produced by the same firm
                 = 0, otherwise
    """

    J = len(prices)
    derivs = simulate_derivs(delta, beta, prices, chars, P0, NS)
    
    Delta  = np.zeros((J,J))
    for firm in ownership:
        for j in firm:
            for q in firm:
                Delta[j][q] = -derivs[j][q]
    
    return Delta

@jit(float64[:](float64[:,:], float64[:], float64[:], float64[:,:], int16[:,:], int16))
def find_markups(mktData, delta, beta, P0, ownership, NS):
    """This function computes J-vector of markups"""

    J = len(mktData)
    K = len(P0[0])-1
    
    prices = np.zeros(J)
    shares = np.zeros(J)
    chars  = np.zeros((J,K))
    
    for j in range(J):
        prices[j] = mktData[j][1]
        shares[j] = mktData[j][2]
        for k in range(K):
            chars[j][k] = mktData[j][3+k]
    
    Delta    = make_Delta(delta, prices, chars, beta, P0, ownership, NS)
    
    invD = np.linalg.inv(Delta)
    return np.dot(invD, shares)


@jit(float64[:](float64[:,:], float64[:], int16, float64, int16))
def find_delta(mktData, P0, NS, tol=1e-8, max_iter=500):
    """Takes mktData (i.e. Data[t][mkt]),
       P0[i][K+1] 
    Returns deltas
    """
    #Num Products in this market
    J = len(mktData)
    #Num prod chars
    K = len(P0[0])-1

    prices = np.zeros(J) 
    shares = np.zeros(J)
    chars  = np.zeros((J,K)) 
    
    for j in range(J):
        prices[j] = mktData[j][1]
        shares[j] = mktData[j][2]
        for k in range(K):
            chars[j][k] = mktData[j][3+k]
    lnShares = np.log(shares)

    #Nevo initial guess:
    delta0 = np.log(shares) - np.log(shares[0])
    delta1 = np.ones(J)

    itr   = 0
    dist = 1

    while dist > tol and itr < max_iter:
        itr += 1
        s_shares = simulate_shares(delta0, prices, chars, P0, NS)

        delta1 = (delta0 + lnShares  
                  - np.log(s_shares))
        
        dist = np.linalg.norm(delta0 - delta1)
        delta0 = delta1 

    return delta0





