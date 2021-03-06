"""This file contains "Estimator" class
Its purpose is to estimate the structural parameters and recover the demand/cost shocks
for each market"""

import numpy as np
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as gmm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv

from scipy.optimize import minimize
from BLPTools import find_delta, find_markups


def convert_to_floats(A): 
    """Converts 2d input array to floats"""
    B = [ [float(e) for e in row] for row in A]
    return B

class Estimator:
    """Takes:
    1) Data (see Market class for details)
    2) K : integer number of product chars
    3) C : integer number of cost chars
    4) N : integer number of simulation iterations
    """
    
    def __init__(self, Data, K, C, N = 500):
        self.Data = convert_to_floats(Data)
        
        

        #Num. Time periods
        self.T = int(max(ob[0] for ob in self.Data)) + 1
        #Num. Markets
        self.M = int(max(ob[1] for ob in self.Data)) + 1
        #Num. Regions
        self.R = int(max(ob[4] for ob in self.Data)) + 1
        #Num Firms
        self.F = int(max(ob[3] for ob in self.Data)) + 1
        #Total number of products
        self.J = int(max(ob[2] for ob in self.Data)) + 1

        #Num Demand-relevant chars
        self.K = K
        #Num Cost-Relevant Chars
        self.C = C

        #num Iterations
        self.N = N

        #itr keeps track of how many times we've evaluated the obj function
        self.itr = 0 

        #We fold data into data[t][mkt][j] = [j, p, s, prod chars, cost chars]
        #Note that the index may be different from the 'j', which denotes ownership
        new_Data = [ [ [] for m in range(self.M)] for t in range(self.T)]
        #prices = [ [-1 for m in range(self.M)] for t in range(self.T)]
        #shares = [ [-1 for m in range(self.M)] for t in range(self.T)]

        ownership = [set() for f in range(self.F)]
        regions   = [set() for r in range(self.R)]
        print(self.T, self.M)
        print(len(new_Data), len(new_Data[0]))

        for obs in self.Data:
            t = int(obs[0])
            m = int(obs[1])
            j = int(obs[2])
            
            new_Data[t][m].append([j]) 
            new_Data[t][m][-1].extend(obs[5:7+self.K+self.C])
            
            ownership[int(obs[3])].add(j)
            regions[int(obs[4])].add(m)
            
        self.Data = new_Data

        self.ownership = [ list(ownership[f]) for f in range(self.F)]
        self.regions   = [ list(regions[r]) for r in range(self.R)]

        #Underlying normal taste shocks
        self.normals = np.random.normal(size=(self.T, self.M,
                                               self.N, self.K+1))

        self.Z = np.array(self.make_IVs())

        self.mZ = np.matrix(np.vstack((self.Z, self.Z)))

        self.invPhi = np.linalg.inv(self.mZ.T*self.mZ)
        
        self.ZinvPhiZT = self.mZ*self.invPhi*self.mZ.T


    def make_z(self, t, mkt):
        """Takes a time and market,
        Returns Z[j] = IVs for products j in that market
        """
        


        z = [ [0 for i in range(1 + 3*self.K + 3*self.C)] 
                for j in range(len(self.Data[t][mkt]))]
        
        #Find region that mkt is in:
        for region in self.regions:
            if mkt in region:
                R = region
                break
        #Compute average prices in region
        count = 0 
        regional_prices = []
        for m in R:
            regional_prices.append(self.Data[t][m][0][1])
            if m == mkt:
                continue
            else:
                count += 1
                for j in range(len(z)):
                       
                    z[j][0] += self.Data[t][m][j][1]
        print("STD ", mkt, " = ", np.std(regional_prices))
        for j in range(len(z)):
            z[j][0] /= (len(R)-1)
        
        print(np.array(regional_prices) - self.Data[t][mkt][0][1], self.Data[t][mkt][0][1]) 

        
        #Compute BLP IVs... Start with finding Firm
        F = [[] for j in range(len(z))]
        for firm in self.ownership:
            for j in range(len(z)):
                if self.Data[t][mkt][j][0] in firm:
                    F[j] = firm
        
        for j in range(len(z)):
            for q in range(len(z)):
                for k in range(self.K):
                    if j == q:
                        z[j][1 + 3*k] += self.Data[t][mkt][q][3+k]
                    elif q in F[j]:
                        z[j][2 + 3*k] += self.Data[t][mkt][q][3+k]
                    else:
                        z[j][3 + 3*k] += self.Data[t][mkt][q][3+k]
                for c in range(self.C):
                    if j == q:
                        z[j][1 + 3*self.K + 3*c] += self.Data[t][mkt][q][3+self.K+c]
                    elif q in F[j]:
                        z[j][2 + 3*self.K + 3*c] += self.Data[t][mkt][q][3+self.K+c]
                    else:
                        z[j][3 + 3*self.K + 3*c] += self.Data[t][mkt][q][3+self.K+c]
        return z


    def make_IVs(self):
        """Produces array of exogenous instruments"""
        Z = []
        for t in range(self.T):
            for mkt in range(self.M):
                Z.extend(self.make_z(t,mkt))
        return Z
#        return sm.add_constant(Z)


    def find_demand_unobs(self, full_deltas):
        """This function computes mean tastes (beta) and demand unobs (xi)
        """

        Y = []
        X = []

        for t in range(self.T):
            for mkt in range(self.M):
                Y.extend(full_deltas[t][mkt])
                for prod in self.Data[t][mkt]:
                    x = np.append(prod[3:3+self.K], prod[1])
                    X.append(x)
        X = sm.add_constant(X)
  
        show_stage1 = True 
        if show_stage1:
            P = [x[-1] for x in X]
            stage1 = sm.OLS(P, self.Z).fit()
            print()
            print("Stage 1:")
            print(stage1.summary())
            print()

        
        reg = gmm.IV2SLS(Y,X,self.Z).fit()
        print(reg.summary())
        beta = reg.params
        xi   = reg.resid

        return beta, xi

    def find_cost_unobs(self, full_deltas, theta, beta):
        B = []
        P = []
        X = []

        P0=np.array([[[[self.normals[t][mkt][i][k]*theta[k] for k in range(self.K+1)]
                    for i in range(self.N)]
                    for mkt in range(self.M)]
                    for t in range(self.T)])
        for t in range(self.T):
            for mkt in range(self.M):
                B.extend(find_markups(self.Data[t][mkt], full_deltas[t][mkt], beta, P0[t][mkt], self.ownership, self.N))
                for j in range(len(self.Data[t][mkt])):
                    P.append(self.Data[t][mkt][j][1])
                    for c in range(self.C):
                        X.append(self.Data[t][mkt][j][3+self.K+c])



        X = sm.add_constant(X)

        Y = np.zeros((len(B)))
        num_bad = 0

        for i in range(len(Y)):
            if P[i] > B[i]:
                Y[i] = P[i] - B[i]
            else:
                num_bad += 1
                Y[i] = 1e-6
        print("NaNs: ", num_bad)
        Y = np.log(Y)
        
        reg = sm.OLS(Y,X).fit()
        print()
        print("Cost Reg")
        print(reg.summary())
        print()
        gamma = reg.params
        om    = reg.resid

        return gamma, om


    def find_Gj(self, theta):
        """Takes STD params, returns objective function"""
        
        #Out of bounds!
        if min(theta) < 0:
            print("Out of bounds")
            return 1e6

        self.itr += 1

        P0 = np.array([[[[self.normals[t][mkt][i][k]*theta[k] for k in range(self.K+1)]
                        for i in range(self.N)]
                        for mkt in range(self.M)]
                        for t in range(self.T)])

        full_deltas = [ [ find_delta(self.Data[t][mkt], P0[t][mkt], self.N) 
                            for mkt in range(self.M)]
                          for t in range(self.T)]
        beta,  xi = self.find_demand_unobs(full_deltas)
        gamma, om = self.find_cost_unobs(full_deltas, theta, beta)

        print()
        print("Mean Xi: ", np.mean(xi))
        print("Std  Xi: ", np.std(xi))
        print()
        print("Mean Om: ", np.mean(om))
        print("Std  Om: ", np.std(om))
        plotter = False 
        if plotter:
            xiSig = 1
            omSig = 0.25

            print()
            print("Mean Xi: ", np.mean(xi))
            print("Std  Xi: ", np.std(xi))
            print()
            print("Mean Om: ", np.mean(om))
            print("Std  Om: ", np.std(om))

            n, bins, patches = plt.hist(xi, 50, normed=1)
            y = mlab.normpdf(bins, 0, xiSig)
            plt.plot(bins, y, 'r--')
            plt.title("Distribution of xi's")
            plt.xlabel("Xi")
            plt.ylabel("Frequency")
            plt.show()
            plt.clf()

            n, bins, patches = plt.hist(om, 50, normed=1)
            y = mlab.normpdf(bins, 0, omSig)
            plt.plot(bins, y, 'r--')
            plt.title("Distribution of om's")
            plt.xlabel("Om")
            plt.ylabel("Frequency")
            plt.show()
            plt.clf()
            plt.close("all")
            

        W = np.matrix(np.concatenate((xi, om)))

        obj = np.linalg.norm(W*self.ZinvPhiZT*W.T)
        print("++++++++")
        print("Where we are: BFGS")
        print(self.itr, ": ", theta, beta, gamma, " : ", obj)
        print("++++++++")
        obj = np.linalg.norm(theta - [0.2, 1])
        return obj

    def estimate(self, guess):
        
        minimum = minimize(self.find_Gj, guess, method="BFGS")
        thetahat = minimum.x

        P0 = np.array([[[[self.normals[t][mkt][i][k]*thetahat[k] for k in range(self.K+1)]
                                    for i in range(self.N)]
                                    for mkt in range(self.M)]
                                    for t in range(self.T)])

        full_deltas = [[ find_delta(self.Data[t][mkt], P0[t][mkt], self.N)
                         for mkt in range(self.M)]
                         for t in range(self.T)]

        
        beta,  xi = self.find_demand_unobs(full_deltas)
        gamma, om = self.find_cost_unobs(full_deltas, thetahat, beta)
        
        print('Theta: ', thetahat, 'Beta: ', beta[1:], 'Gamma: ', gamma[1:])
        self.write_estimates(thetahat, beta[1:], gamma[1:], xi, om, full_deltas, P0)

    def write_estimates(self, theta, beta, gamma, xi, om, full_deltas, P0):
        with open('xi.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(xi)
        with open('om.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(om)
        with open('estimates.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(theta)
            writer.writerow(beta)
            writer.writerow(gamma)
        
        """Write table with markups"""
        B = [ [] for j in range(self.J)] #Percent Markups
        P = [ [] for j in range(self.J)] #Prices
        C = [ [] for j in range(self.J)] #Marginal Costs
        for t in range(self.T):
            for m in range(self.M):
                mktData = self.Data[t][m]
                b = find_markups(mktData, full_deltas[t][m], beta, P0, self.ownership, self.N)

                for j in range(len(mktData)):
                    i = mktData[j][0]
                    B[i].append(b[j]/mktData[j][1]) 
                    P[i].append(mktData[j][1])
                    C[i].append(mktData[j][1]-b[j])

        
        med_B = [ np.median(B[j]) for j in range(self.J)]
        med_P = [ np.median(P[j]) for j in range(self.J)]
        med_C = [ np.median(P[j]) for j in range(self.J)]

        table = [ [med_P[j], med_C[j], med_B[j]] for j in range(self.J)]
        np.savetxt("markups_table.txt", table, delimiter=' & ',
                    fmt='%2.2f', newline=' \\\\ \\hline \n')

        """Write table with elasticities"""
        E = [ [ [] for q in range(self.J)] for j in range(self.J)]
        for t in range(self.T):
            for m in range(self.M):
                mktData = self.Data[t][m]
                prices  = [mktData[j][1] for j in range(len(mktData))]
                shares  = [mktData[j][2] for j in range(len(mktData))]
                pChars  = [mktData[j][3:3+self.K] for j in range(len(mktData))]

                d = simulate_derivs(full_deltas[t][m], beta, prices, pChars, P0, self.N)
                e = [ [ d[j][q]*prices[q]/shares[j] for q in range(len(d))] for j in range(len(d))]

                for j in range(len(mktData)):
                    for q in range(len(mktData)):
                        k = mktData[j][0]
                        r = mktData[q][0]
                        E[k][r].append(e[j][q])
        table = [ [np.median(E[j][q]) for q in range(self.J)] for j in range(self.J)]
        np.savetxt("elasticities_table.txt", table, delimiter=' & ',
                    fmt='%2.2f', newline=' \\\\ \\hline \n')






