import numpy as np
import pandas as pd
import pickle
from multiprocessing import pool


data=pd.read_csv('all-prediction-matrix.csv').values #global matrix accessed by all treads. bad, bad, bad coding practice, but research
def compute_disagreement_row_in_upper_triangular(i): #i is the reference column to compute disagreement with
    right_results= [np.logical_xor(data[:, i], data[:, j]).sum() for j in range(i+1,data.shape[1])] #only compute for columns on the right of i
    results = np.zeros(data.shape[1])
    results[i+1:] = right_results[:] #pad with zeros on the left side
    return i, results

if __name__ == "__main__":
    poo = pool.Pool()
    res = poo.map(compute_disagreement_row_in_upper_triangular, range(data.shape[1]))
    results = np.vstack([j for i, j in sorted(res, key=lambda x:x[0])]) #sort and combine rows
    results += results.T #copy upper triangular to lower triangular
    results = results/float(data.shape[0]) #the fraction of disagreements
    results = 1.-results #the fraction of agreements

    pickle.dump(results, open( "results.p", "wb" ) )
