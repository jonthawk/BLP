from estimation import Estimator
import numpy as np
import csv


A = [0, 1, 2, 3]
print(0 in A)


def main():
    np.random.seed(0)
    Data = []
    with open("test_data_0.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            Data.append(row)

    foo = Estimator(Data, 1, 1, N=1000)
    foo.estimate([0.2, 1.0])   
    print("Test Data 0") 

main()
