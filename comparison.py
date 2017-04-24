import numpy as np
import csv

"""This compares pre-merger, post-merger, and counterfactual
    1) mean/median prices before, after, and counterfactual
    2) predicted changes vs. observed changes
    3) plots distribution of prices before/after for selected brands
    
    4) mean/median market shares before, after, and counter

"""

def convert_to_floats(A):
    return [[float(e) for e in row] for row in A]

def clean_mean(A):
    """Takes A[type][mkt][j], where type=0 is premerger, type=1 is postmerger, type=2 is prediction
       Returns B[type][j] = means across markets WHERE j IS PRESENT 
    """
    
    sums = [ [0 for j in range(len(A[0][0]))] for i in range(3)]
    cnts = [ [0 for j in range(len(A[0][0]))] for i in range(3)]
    for i in range(3):
        for m in range(len(A[0])):
            for j in range(len(A[0][0])):
                if A[i][m][j] == 0:
                    continue
                else:
                    sums[i][j] += A[i][m][j]
                    cnts[i][j] += A[i][m][j]
    for i in range(3):
        for j in range(len(A[0][0])):
            sums[i][j] = sums[i][j]/cnts[i][j]
    return sums

def clean_median(A):
    """Takes same input as clean_mean,
       Returns B[type][j] = median across markets where j is present
    """
    
    #Make cln_A[type][j][mkt] (without markets where 0)
    cln_A = [ [ [] for j in range(len(A[0][0]))] for i in range(3)]

    for i in range(3):
        for m in range(len(A[0])):
            for j in range(len(A[0][0])):
                if A[i][m][j] == 0:
                    continue
                else:
                    cln_A[i][j].append(A[i][m][j])
    
    return [ [np.median(cln_A[i][j]) for j in range(len(A[0][0]))] for i in range(3)]



P_Data = []
C_Data = []
R_Data = []
#Pre-merger data (used for estimation)
with open('real_data_0.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        R_Data.append(row)

#Post-merger data (true impact of merger)
with open('real_data_0.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        P_Data.append(row)

#Counterfactual data (post-merger ownership, pre-merger estimates)
with open('counterfactual_data_0.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        C_Data.append(row)

R_Data = convert_to_floats(R_Data)
P_Data = convert_to_floats(P_Data)
C_Data = convert_to_floats(C_Data)

rT = int(max(ob[0] for ob in R_Data)) + 1
pT = int(max(ob[0] for ob in P_Data)) + 1
cT = int(max(ob[0] for ob in C_Data)) + 1

M  = int(max(ob[1] for ob in R_Data)) + 1
J  = int(max(ob[2] for ob in R_Data)) + 1


nR_Data = [ [ [] for m in range(M)] for t in range(rT)]
nC_Data = [ [ [] for m in range(M)] for t in range(cT)]
nP_Data = [ [ [] for m in range(M)] for t in range(pT)]

for obs in R_Data:
    t = int(obs[0])
    m = int(obs[1])
    j = int(obs[2])

    nR_Data[t][m].append([j])
    nR_Data[t][m][-1].extend(obs[5:7])

R_Data = nR_Data

for obs in C_Data:
    t = int(obs[0])
    m = int(obs[1])
    j = int(obs[2])
                    
    nC_Data[t][m].append([j])
    nC_Data[t][m][-1].extend(obs[5:7])                           

C_Data = nC_Data

for obs in P_Data:
    t = int(obs[0])
    m = int(obs[1])
    j = int(obs[2])

    nP_Data[t][m].append([j])
    nP_Data[t][m][-1].extend(obs[5:7])

P_Data = nP_Data


prices = [[ [0 for j in range(J)] for m in range(M)] for i in range(3)]
shares = [[ [0 for j in range(J)] for m in range(M)] for i in range(3)]

for m in range(M):
    for t in range(rT):
        for obs in R_Data[t][m]:
            prices[0][m][obs[0]] += obs[1]/rT
            shares[0][m][obs[0]] += obs[2]/rT
    for t in range(pT):
        for obs in P_Data[t][m]:
            prices[1][m][obs[0]] += obs[1]/pT
            shares[1][m][obs[0]] += obs[2]/pT
    for t in range(cT):
        for obs in C_Data[t][m]:
            prices[2][m][obs[0]] += obs[1]/cT
            shares[2][m][obs[0]] += obs[2]/cT


p_means = clean_mean(prices)
s_means = clean_mean(shares)

p_meds = clean_median(prices)
s_meds = clean_median(shares)

#Predicted percent change in medians (c-r)/r, PPC[j]=[price, quantity]


Table = [ [ p_meds[0][j], s_meds[0][j], #Pre-Merger
            p_meds[2][j], s_meds[2][j], #Predicted post-merger
            p_meds[1][j], s_meds[1][j], #Observed post-merger
            (p_meds[2][j] - p_meds[0][j])/p_meds[0][j],
            (s_meds[2][j] - s_meds[0][j])/s_meds[0][j], #Predicted change
            (p_meds[1][j] - p_meds[0][j])/p_meds[0][j],
            (s_meds[1][j] - s_meds[0][j])/s_meds[0][j]]
        for j in range(J)]

np.savetxt("counterfactual_table.txt", Table, delimiter=' & ', 
            fmt='%2.2f', newline=' \\\\ \\hline  \n')



