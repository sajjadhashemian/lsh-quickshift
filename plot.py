import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_data(file_path):
	with open(file_path, 'r') as file:
		lines = file.readlines()
    
	n_rows = len(lines)-1
	D = len(lines[0].split())
	data_points = []
	for i in range(n_rows):
		data_points.append(list(map(float, lines[i].split())))
	data_points = np.array(data_points)
    
	cluster_ids=list(map(int, lines[len(lines)-1].split()))
	cluster_ids = np.array(cluster_ids)

	return data_points, cluster_ids

def plot_clusters(data_points, cluster_ids):
	unique_clusters = np.unique(cluster_ids)
	colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_clusters))))

	plt.figure(figsize=(8, 6))
	for cluster in unique_clusters:
		cluster_data = data_points[cluster_ids == cluster]
		plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=next(colors), label=f'Cluster {cluster}: {cluster_data.shape[0]}')
    
	plt.title('Clustered Data Points')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()
	plt.show()

file_path = 'out.txt'
data_points, cluster_ids = read_data(file_path)
print('done')
print(cluster_ids)
plot_clusters(data_points, cluster_ids)