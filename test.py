import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from copy import copy
from time import time
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_digits

from quickshift import QuickShift
from cpp import qs

# Function to apply clustering algorithms and evaluate results
def cluster_and_evaluate(file_paths, true_labels_column, clustering_algorithms, output_csv, __dir):
    # Initialize a DataFrame to store results

    for file_path in file_paths:
        # Read the dataset
        # data = pd.read_csv(file_path)

        # Separate features and true labels
        # X = data.drop(columns=[true_labels_column])
        # true_labels = data[true_labels_column]
        X = file_path.data
        true_labels = file_path.target

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for algo_name, algo in clustering_algorithms.items():
            results_df = pd.DataFrame(columns=["Dataset", "Algorithm", "AMI", "ARI", "Time"])

            t1 = time()
            if(algo_name=="KMeans"):
                clustering = algo(n_clusters=len(np.unique(true_labels))).fit(X_scaled)
            elif(algo_name=="DBSCAN"):
                _eps = estimate_bandwidth(X_scaled, n_samples=min(1000,len(X_scaled)))
                clustering = algo(eps = _eps).fit(X_scaled)
            else:
                _eps = estimate_bandwidth(X_scaled, n_samples=min(1000,len(X_scaled)))
                clustering = algo(bandwidth = _eps).fit(X_scaled)

            t2 = time()-t1
            predicted_labels = clustering.labels_

            # Calculate AMI and ARI
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            ari = adjusted_rand_score(true_labels, predicted_labels)

            # Append results to the DataFrame
            res = pd.read_csv(__dir+algo_name+output_csv)
            results_df = pd.concat([res, pd.DataFrame({
                "Dataset": [file_path],
                "Algorithm": [algo_name],
                "AMI": [ami],
                "ARI": [ari],
                "Time": [t2]
            })], ignore_index=True)

            # Save the results to a CSV file
            results_df.to_csv(__dir+algo_name+output_csv, index=False)
            print(f"Results saved to {algo_name+output_csv}")

# Example usage
if __name__ == "__main__":
    # List of CSV file paths
    file_paths = [
        load_iris(),
        load_wine(),
        load_digits(),
    ]

    # Column name for true labels
    true_labels_column = "true_labels"

    # Define clustering algorithms
    clustering_algorithms = {
        "KMeans": KMeans(random_state=42),
        "DBSCAN": DBSCAN(min_samples=5),
        "MeanShift": MeanShift(),
        "QuickShift": QuickShift()
    }

    # Output CSV file to save results
    output_csv = "_clustering_results.csv"
    result_dir = "./results/"

    # Apply clustering and evaluate results
    cluster_and_evaluate(file_paths, true_labels_column, clustering_algorithms, output_csv, result_dir)