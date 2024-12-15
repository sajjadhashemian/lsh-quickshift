import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from copy import copy
from time import time
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from cpp import qs

# Function to apply clustering algorithms and evaluate results
def cluster_and_evaluate(file_paths, true_labels_column, clustering_algorithms, output_csv):
    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=["Dataset", "Algorithm", "AMI", "ARI"])

    for file_path in file_paths:
        # Read the dataset
        data = pd.read_csv(file_path)

        # Separate features and true labels
        X = data.drop(columns=[true_labels_column])
        true_labels = data[true_labels_column]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for algo_name, algo in clustering_algorithms.items():
            # Apply clustering algorithm
            if algo_name == "DBSCAN":
                # DBSCAN does not require specifying the number of clusters
                clustering = algo.fit(X_scaled)
                predicted_labels = clustering.labels_
            else:
                clustering = algo.fit(X_scaled)
                predicted_labels = clustering.labels_

            # Calculate AMI and ARI
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            ari = adjusted_rand_score(true_labels, predicted_labels)

            # Append results to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame({
                "Dataset": [file_path],
                "Algorithm": [algo_name],
                "AMI": [ami],
                "ARI": [ari]
            })], ignore_index=True)

    # Save the results to a CSV file
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    # List of CSV file paths
    file_paths = [
        "dataset1.csv",
        "dataset2.csv",
        "dataset3.csv"
    ]

    # Column name for true labels
    true_labels_column = "true_labels"

    # Define clustering algorithms
    clustering_algorithms = {
        "KMeans": KMeans(n_clusters=3, random_state=42),
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
    }

    # Output CSV file to save results
    output_csv = "clustering_results.csv"

    # Apply clustering and evaluate results
    cluster_and_evaluate(file_paths, true_labels_column, clustering_algorithms, output_csv)