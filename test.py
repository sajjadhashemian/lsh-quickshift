import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from math import log
from copy import copy
from time import time

from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_digits

from quickshift import QuickShift
from fastqs import _lsh_quickshift

class DATA:
    def __init__(self, name, data, target):
        self.name = name
        self.data = data
        self.target = target

def __evaluate(dataset, clustering_algorithms, output_csv):
    # Initialize a DataFrame to store results

    for i, data in enumerate(dataset):
        X = data.data
        true_labels = data.target

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for algo_name, algo in clustering_algorithms.items():
            results_df = pd.DataFrame(columns=["Dataset", "Algorithm", "AMI", "ARI", "Time"])

            t1 = time()
            if(algo_name=="KMeans"):
                clustering = algo(n_clusters=len(np.unique(true_labels))).fit(X_scaled)
            elif(algo_name=="DBSCAN"):
                if(data.name[0]=='m'):
                    continue
                _eps = estimate_bandwidth(X_scaled, n_samples=min(1000,len(X_scaled)))
                clustering = algo(eps = _eps, min_samples=int(X.shape[0]**0.5)).fit(X_scaled)
            elif(algo_name=="LSH-QuickShift"):
                clustering = algo(X_scaled, c=2)
            else:
                if(data.name[0]=='m'):
                    continue
                _eps = estimate_bandwidth(X_scaled, n_samples=min(1000,len(X_scaled)))
                clustering = algo(bandwidth = _eps).fit(X_scaled)
            t2 = time()-t1

            if(algo_name!="LSH-QuickShift"):
                predicted_labels = clustering.labels_
            else:
                predicted_labels = clustering
            
            # Calculate AMI and ARI
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            ari = adjusted_rand_score(true_labels, predicted_labels)

            # Append results to the DataFrame
            try:
                res = pd.read_csv(output_csv)
            except:
                res = pd.DataFrame(columns=["Dataset", "n", "D", "C", "Algorithm", "AMI", "ARI", "Time"])
            results_df = pd.concat([res, pd.DataFrame({
                "Dataset": [data.name],
                "n": [X.shape[0]],
                "D": [X.shape[1]],
                "C": [len(np.unique(true_labels))],
                "Algorithm": [algo_name],
                "AMI": [ami],
                "ARI": [ari],
                "Time": [t2]
            })], ignore_index=True)

            results_df.to_csv(output_csv, index=False)
            print(f"{data.name} results on {algo_name} is saved to {output_csv}.")



def read_csv(__dir):
    data = []
    for file_name in glob.glob(__dir+'*.csv'):
        x = pd.read_csv(file_name)
        X = (x.iloc[:,:-1]).to_numpy()
        y = (x.iloc[:,-1:]).to_numpy()
        u, y = np.unique(y, return_inverse=True)
        data.append(DATA(file_name[7:-4], X, y))

    for file_name in glob.glob(__dir+'*.t'):
        x = pd.read_csv(file_name, sep='\t')
        X = (x.iloc[:,:-1]).to_numpy()
        y = (x.iloc[:,-1:]).to_numpy()
        u, y = np.unique(y, return_inverse=True)
        data.append(DATA(file_name[7:-4], X, y))
    return data



if __name__ == "__main__":
    data = [
        DATA('iris', load_iris().data, load_iris().target),
        DATA('digits', load_digits().data, load_digits().target),
    ]
    data += read_csv('./data/')

    # Define clustering algorithms
    clustering_algorithms = {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN,
        "MeanShift": MeanShift,
        "QuickShift": QuickShift,
        "LSH-QuickShift": _lsh_quickshift
    }

    # Output CSV file to save results
    output_suffix = "results.csv"
    
    print("HERE")

    # Apply clustering and evaluate results
    __evaluate(data, clustering_algorithms, output_suffix)