"""This file contains Market class.
   It takes an array of parameters and structural errors
   It generates share/price data for verifying estimation procedure"""


import numpy as np
from numba import jit

class Market:
    """A market is initialized with:
    1) theta: a non-linear parameter array
              contains standard dev. of taste params
              theta[0]  = price-sensitivity
              theta[1:] = betas

    2) beta:  a linear parameter array
              beta[k] = mean taste for prod_char. k

    3) gamma: a linear parameter array
              gamma[k] = cost shifting param for cost_char. k

    4) ownership: an array of lists
              each list contains product indices produced by a firm
              ownership[f] = list of product indices produced by firm
              ownership is indexed starting at i, as produce 0 is OO
             
    5) regions: an array of lists
              each list contains market indices for that region
              regions[r] = list of mkt indices contained in that region

    6) prod_chars: a JxK array
              prod_chars[j][k] = value of char. k for prod. j

    7) cost_chars: a JxC array
              cost_chars[j][c] = value of cost shifter c for prod. j

    8) om: a J-array
              Contains cost shocks for each product
    
    9) xi: a J-array
              Contains quality unobservables for each product

    10) N: an integer
              Number of "individuals" in each market simulation

    """


    def __init__(self, theta, beta, gamma,
                 ownership, regions, 
                 prod_chars, cost_chars,
                 om, xi, N=1000):
        self.theta = theta
        self.beta  = beta
        self.gamma = gamma

        self.ownership  = ownership
        self.regions    = regions
        self.prod_chars = prod_chars
        self.cost_chars = cost_chars
        
        self.N = N

        #Num. products (sauf OO)
        self.J = len(prod_chars)
        #Num prod chars
        self.K = len(prod_chars[0])

        self.xi = xi
        self.om = om
        
        self.MC = np.exp(om + np.dot(self.cost_chars, self.gamma)) 

        self.P0 = np.transpose(
                    np.array([
                        theta[k]*np.random.normal(size=self.N) 
                        for k in range(self.K+1)]))


    def find_firm(self, j):
        """Returns index of firm producing j"""

        for i, firm in enumerate(self.ownership):
            if j+1 in firm:
                return i
        print("ERROR: Product ", j, " not found")
        return None

    def find_region(self, mkt):
        """Returns index of region mkt is in"""
        for i, region in enumerate(self.regions):
            if mkt in region:
                return i
        print("ERROR: Market ", mkt, " not found")
        return None


    def simulate_shares(self, prices):
        """Takes an array of J-prices,
           Return an array of J product shares
           """

        totals = np.zeros(self.J)
        for i in range(self.N):
            nu = self.P0[i]
            beta_i = [self.beta[k+1] + nu[k+1] for k in range(self.K)]
            expD   = np.exp([np.dot(self.prod_chars[j], beta_i)
                             - (self.beta[0]+nu[0])*prices[j]
                             + self.xi[j] for j in range(self.J)])
            denom = 1 + np.sum(expD)
            for j in range(self.J):
                totals[j] += expD[j]/denom
        
        return totals/self.N

    def simulate_derivs(self, prices):
        """Takes an array of J prices
        Returns a JxJ matrix of share/price derivatives
        """

        totals = np.zeros((self.J, self.J))
        for i in range(self.N):
            nu = self.P0[i]
            beta_i = [self.beta[k+1]+nu[k+1] for k in range(self.K)]
            expD = np.exp([np.dot(self.prod_chars[j], beta_i)
                            - (self.beta[0]+nu[0])*prices[j]
                            + self.xi[j] for j in range(self.J)])
            denom = 1 + np.sum(expD)

            F = [expD[j]/denom for j in range(self.J)]

            Dmu = [-(self.beta[0]+nu[0]) for j in range(self.J)]

            for j in range(self.J):
                for q in range(j+1):
                    if j == q: 
                        totals[j][q] += F[j]*(1-F[j])*Dmu[j]
                    else:
                        totals[j][q] += -F[j]*F[q]*Dmu[q]

        for j in range(self.J):
            for q in range(j):
                totals[q][j] = totals[j][q] 
        
        return totals/self.N

    def choose_prices(self, prices):
        """Takes array of J prices
        Returns array of J optimal prices (BR to input prices)
        """

        derivs = self.simulate_derivs(prices)
        Delta  = np.zeros((self.J, self.J))
        for firm in self.ownership:
            for j in firm:
                for r in firm:
                    Delta[j-1][r-1] = -derivs[j-1][r-1]

        D = np.linalg.inv(Delta)
        s = self.simulate_shares(prices)

        return self.MC + np.dot(D,s)

    def equilibrium(self, tol=1e-6, maxiter=1000):
        """Returns Nash EQ prices for this market"""
    
        prices0 = np.ones(self.J)
        prices1 = prices0

        i = 0
        
        diff = 100
        dif0 = 100
        adj  = 1

        while diff > tol and i < maxiter:
            i += 1
            prices1 = self.choose_prices(prices0)
            
            diff = np.linalg.norm(prices0 - prices1)
            prices0 = adj*prices1 + (1-adj)*prices0

            if i % 10 == 0:
                print(i, " : ", diff, "Adj: ", adj)

            if diff/dif0 >= 0.7:
                adj = adj*0.9
            dif0 = diff

        shares = self.simulate_shares(prices0)
        return prices0, shares

    def produce_data(self, t, mkt):
        """Returns rows of data for this market"""

        prices, shares = self.equilibrium()

        print()
        print(t,mkt)
        print("Prices: ")
        print(prices)
        print("Shares: ")
        print(shares, " OO: ", 1-np.sum(shares))
        print()

        Data = [[t,mkt,j,self.find_firm(j), self.find_region(mkt),
                prices[j], shares[j]] for j in range(self.J)]
        for j in range(self.J):
            Data[j].extend(self.prod_chars[j])
            Data[j].extend(self.cost_chars[j])
            Data[j].append(self.om[j])
            Data[j].append(self.xi[j])

        return Data
