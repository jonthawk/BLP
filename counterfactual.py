import numpy as np
import csv

import Market from market_class.py


def convert_to_floats(A):
    return [[float(e) for e in row] for row in A]


#Num. Demand Chars
K = 1
#Num Cost Chars
C = 1

#Counterfactual Ownership
ownership = [ [1,2,3,4,5,6],
              [7,8,9,10]]

xi = []
om = []

estimates = []
theta = []
beta  = []
gamma = []

Data  = []

with open('xi.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        xi.append(row)
with open('om.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        om.append(row)
with open('estimates.csv', 'r') as f:
    reader = csv.reader
    for row in reader:
        estimates.append(row)
with open('real_data.csv', 'r') as f:
    reader = csv.reader:
    for row in reader:
        Data.append(row)

xi   = convert_to_floats(xi)
om   = convert_to_floats(om)
est  = convert_to_floats(estimates)
Data = convert_to_floats(Data)

theta = est[0]
beta  = est[1]
gamma = est[2]

#Number of Time Periods, markets, regions, firms
T = int(max(ob[0] for ob in Data)) + 1
M = int(max(ob[1] for ob in Data)) + 1
R = int(max(ob[4] for ob in Data)) + 1
F = len(ownership)

new_Data = [ [ [] for m in range(M)] for t in range(T)]
regions = [set() for r in range(R)]

for obs in Data:
    t = int(obs[0])
    m = int(obs[1])
    j = int(obs[2])

    new_Data[t][m].append([j])
    new_Data[t][m][-1].extend(obs[5:7+K+C])

    regions[int(obs[4])].add(m)

Data = new_Data
regions = [ list(regions[r]) for r in range(R)]

counter_data = []

place0 = 0
place1 = 0
for t in range(T):
    for m in range(M):
        place1 += len(Data[t][m])
        p_char = Data[t][m][3:3+K]
        c_char = Data[t][m][3+K:3+K+C]
        
        print("Computing: ", (t, m), "...")
        foo = Market(theta, beta, gamma,
                    ownership, regions, p_char, c_char
                    xi[place0:place1], om[place0:place1],
                    N=500)
        counter_data.extend(foo.produce_data(t,mkt))

with open("counterfactual_data.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(counter_data)




