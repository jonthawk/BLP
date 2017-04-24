from estimation import Estimator
import numpy as np
import csv

def main():
    np.random.seed(0)
    Data = []
    with open("test_data_BIG.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            Data.append(row)

    foo = Estimator(Data, 1, 1, N=1000)
    foo.estimate([0.1, 0.1])   
    print("Big!") 

main()
