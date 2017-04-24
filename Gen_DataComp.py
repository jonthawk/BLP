import numpy as np
import csv

from market_class import Market

def main():
    np.random.seed(0)
    num_prod = 10
    theta = [0.2, 1]
    beta  = [1,   5]
    gamma = [1]


    ownership1 = [ [1,2,3,4],
                   [5,6],
                   [7,8,9],
                   [10]]

    ownership2 = [ [1,2,3,4,5,6],
                   [7,8,9,10]]


    omSig = 0.25
    xiSig = 1

    regions = [range(5*r, 5*(r+1)) for r in range(10)]
    print(regions)
    Data1 = []
    Data2 = []
    for t in range(4):
        for mkt in range(50):
            if mkt % 5 == 0:        
                print(mkt, "Switching Regions") 
                marg_cost = omSig*np.random.normal(size=num_prod)
                prod_chars = np.random.rand(num_prod, 1)
                cost_chars = 0.5*np.random.rand(num_prod, 1)
            xi = xiSig*np.random.normal(size=num_prod)
            print("Computing: ", (t, mkt), "...")
            foo = Market(theta, beta, gamma,
                         ownership1, regions, prod_chars, cost_chars,
                         marg_cost, xi, N=500)
            bar = Market(theta, beta, gamma,
                         ownership2, regions, prod_chars, cost_chars,
                         marg_cost, xi, N=500)

            Data1.extend(foo.produce_data(t,mkt))
            Data2.extend(bar.produce_data(t,mkt))

    with open("real_data_0.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Data1)
    with open("counterfactual_data_0.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Data2)

main()
