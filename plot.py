import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata

def read_data(file_path):
	with open(file_path, 'r') as file:
		lines = file.readlines()
    
	n_rows = len(lines)-3
	D = len(lines[0].split())
	data_points = []
	for i in range(n_rows):
		data_points.append(list(map(float, lines[i].split())))
	data_points = np.array(data_points)
	print('here', data_points.shape)
    
	density = list(map(float, lines[len(lines)-3].split()))
	density = np.array(density)*100
	print('here', density.shape)

	idx = list(map(int, lines[len(lines)-2].split()))
	# for i in idx:
	# 	print(data_points[i])

	cluster_ids = list(map(int, lines[len(lines)-1].split()))
	cluster_ids = np.array(cluster_ids)
	print('here', cluster_ids.shape)

	return data_points, cluster_ids, density, idx

def plot_clusters(data_points, cluster_ids, density, idx):
	# Plot the clustered data points
	unique_clusters = np.unique(cluster_ids)
	colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_clusters))))

	plt.figure(figsize=(16, 8))

	# Subplot 1: Scatter plot of clusters
	plt.subplot(1, 2, 1)
	for cluster in unique_clusters:
		cluster_data = data_points[cluster_ids == cluster]
		plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=next(colors), label=f'Cluster {cluster}: {cluster_data.shape[0]}')
    
	for i in idx:
		plt.scatter(data_points[i, 0], data_points[i, 1], label=f'Core: {data_points[i]}, rho: {density[i]} ')
    
	plt.title('Clustered Data Points')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()

    # Subplot 2: Heat map of density (interpolated)
	plt.subplot(1, 2, 2)
    # Extract x, y, and density values
	x = data_points[:, 0]
	y = data_points[:, 1]
    # Create a grid for interpolation
	grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    # Interpolate density onto the grid
	grid_density = griddata((x, y), density, (grid_x, grid_y), method='cubic')
    # Plot the heat map
	plt.imshow(grid_density.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
	plt.colorbar(label='Density')
	plt.title('Density Heat Map (Interpolated)')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')

	plt.tight_layout()
	plt.show()

# File path
file_path = 'out.txt'
data_points, cluster_ids, density, idx = read_data(file_path)
print('done')
print(cluster_ids)
plot_clusters(data_points, cluster_ids, density, idx)