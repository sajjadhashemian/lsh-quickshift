#include "common.h"
#include "kde.h"

class QuickShift
{
public:
	QuickShift(int k_neighbors) : k(k_neighbors){}
	std::vector<int> fit(const std::vector<std::vector<float>> &data, int n_bits)
	{
		int n = data.size();
		int dim = data[0].size();

		// Prepare data for Faiss
		float *dataset = new float[n * dim];
		for (int i = 0; i < n; ++i)
		{
			for (int d = 0; d < dim; ++d)
			{
				dataset[i * dim + d] = data[i][d];
			}
		}

		// Build Faiss index
		faiss::IndexFlatL2 index(dim); // Flat index for exact nearest neighbor search
		index.add(n, dataset);

		// Compute density for each point using KDE
		KDE kde(dim, n_bits);
		kde.fit(data);

		std::vector<float> densities(n, 0.0);
		std::vector<faiss::idx_t> most_dense_neighbors(n, -1); // Use faiss::idx_t for labels

		// Find k-nearest neighbors for each point
		faiss::idx_t *labels = new faiss::idx_t[k]; // Use faiss::idx_t for labels
		float *distances = new float[k];
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);

			// Compute density using KDE estimator
			std::vector<float> query_point(dim);
			for (int d = 0; d < dim; ++d)
			{
				query_point[d] = data[i][d];
			}
			densities[i] = kde.query(query_point, k);
		}

		// Find the most dense neighbor for each point
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);

			// Finding the most dense neighbor (excluding itself)
			float max_density = -1.0;
			faiss::idx_t most_dense_neighbor = -1;
			for (int j = 0; j < k; ++j)
			{
				faiss::idx_t neighbor = labels[j];
				if (neighbor != i && densities[neighbor] > max_density)
				{
					max_density = densities[neighbor];
					most_dense_neighbor = neighbor;
				}
			}

			// If the current point is the most dense, assign it to itself
			if (densities[i] > max_density)
			{
				max_density = densities[i];
				most_dense_neighbor = i;
			}
			most_dense_neighbors[i] = most_dense_neighbor;
		}

		// Assign clusters based on the most dense neighbors
		std::vector<int> clusters(n, -1);
		int cluster_id = 0;
		for (int i = 0; i < n; ++i)
		{
			faiss::idx_t current = i;
			while (most_dense_neighbors[current] != current)
				current = most_dense_neighbors[current];
			if (clusters[current] == -1)
				clusters[current] = cluster_id++;
		}

		// Propagate cluster assignments
		for (int i = 0; i < n; ++i)
		{
			if (clusters[i] == -1)
			{
				faiss::idx_t current = i;
				faiss::idx_t par = most_dense_neighbors[current];
				while (current != par)
				{
					current = par;
					par = most_dense_neighbors[par];
				}
				clusters[i] = clusters[current];
				most_dense_neighbors[i] = current;
			}
		}

		delete[] dataset;
		delete[] labels;
		delete[] distances;

		return clusters;
	}
private:
	int k; // Number of neighbors for k-NN search
};