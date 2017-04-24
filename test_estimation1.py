from estimation import Estimator
import numpy as np
import csv

def main():
    np.random.seed(0)
    Data = []
    with open("test_data_1.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            Data.append(row)

    foo = Estimator(Data, 1, 1, N=500)
    foo.estimate([0.2, 0.9])   
    print("Test Data 1") 

main()
