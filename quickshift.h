#ifndef QUICKSHIFT_H
#define QUICKSHIFT_H

#include "common.h"
#include "kde.h"

class QuickShift
{
private:
	int k;									// Number of neighbors for k-NN search
	int n_bits;								// Number of bits for LSH
	int dim;								// Dimension of the data
	int n;									// Number of the data

public:
	QuickShift(int k_neighbors) : k(k_neighbors){}
	
	vector<int> fit(const vector<vector<float>> &data, int n_bits = 128)
	{
		n = data.size(); dim = data[0].size();

		// Prepare data for Faiss
		float *dataset = new float[n * dim];
		for (int i = 0; i < n; ++i)
			for (int d = 0; d < dim; ++d)
				dataset[i * dim + d] = data[i][d];

		// Build Faiss index
		faiss::IndexFlatL2 index(dim);
		index.add(n, dataset);

		// Compute density for each point using KDE
		KDE kde(dim, n_bits);
		kde.fit(data);

		vector<float> density(n, 0.0);
		vector<faiss::idx_t> parents(n, -1);

		// Find k-nearest neighbors for each point
		faiss::idx_t *labels = new faiss::idx_t[k];
		float *distances = new float[k];
		float mn=1000;
		float mx=-1;
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);

			// Compute density using KDE estimator
			vector<float> query_point(dim);
			for (int d = 0; d < dim; ++d)
				query_point[d] = data[i][d];
			density[i] = kde.query(query_point, k*2);
			// cout<<density[i]<<" ";
			// mx=max(density[i],mx);
			// mn=min(mn, density[i]);
		}
		// cerr<<mn<<", "<<mx<<endl;
		// cout<<endl;

		// Assign each point to the nearest neighbor with higher density
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);

			// Find the nearest neighbor with highest density
			faiss::idx_t heaviest_neighbor = i;
			float min_distance = numeric_limits<float>::max();
			for (int j = 0; j < k; ++j)
			{
				faiss::idx_t neighbor = labels[j];
				if (density[neighbor] > density[heaviest_neighbor])
				{
					min_distance = distances[j];
					heaviest_neighbor = neighbor;
				}
			}
			parents[i] = heaviest_neighbor;
		}

		// Assign clusters based on the parent structure
		vector<int> clusters(n, -1);
		int cluster_id = 0;
		for (int i = 0; i < n; ++i)
		{
			faiss::idx_t current = i;
			while (parents[current] != current)
				current = parents[current];
			if (clusters[current] == -1)
				// cerr<<current<<": "<<density[current]<<endl,
				// cout<<current<<" ",
				clusters[current] = cluster_id++;
		}
		// cout<<endl;
		// Propagate cluster assignments
		for (int i = 0; i < n; ++i)
			if (clusters[i] == -1)
			{
				faiss::idx_t current = i;
				faiss::idx_t par = parents[current];
				while (current != par)
				{
					current = par;
					par = parents[par];
				}
				clusters[i] = clusters[current];
				parents[i] = current;
			}

		delete[] dataset;
		delete[] labels;
		delete[] distances;

		return clusters;
	}

	vector<int> fast_fit(const vector<vector<float>> &data)
	{
		int n = data.size();
		int dim = data[0].size();

		// sample sqrt(n) vector
		vector<vector<float>> sample;
		vector<int> index_sample;
		for (int i = 0; i < n; i++)
			if (__cointoss(n))
			{
				sample.push_back(data[i]);
				index_sample.push_back(i);
			}
		int m = sample.size();
		int _k = sqrt(m)*log(m);
		// cerr << sample.size() << " " << _k << endl;

		QuickShift qs(_k);
		vector<int> result = qs.fit(sample);
		vector<int> clusters(n, -1);
		for (int i = 0; i < index_sample.size(); ++i)
			clusters[index_sample[i]] = result[i];

		// Prepare data for Faiss
		float *sampleset = new float[m * dim];
		for (int i = 0; i < m; ++i)
			for (int d = 0; d < dim; ++d)
				sampleset[i * dim + d] = data[index_sample[i]][d];
		float *dataset = new float[n * dim];
		for (int i = 0; i < n; ++i)
			for (int d = 0; d < dim; ++d)
				dataset[i * dim + d] = data[i][d];


		// Build Faiss index
		faiss::IndexFlatL2 index(dim); // Flat index for exact nearest neighbor search
		index.add(m, sampleset);

		// Find k-nearest neighbors for each point and Assigning clusters
		faiss::idx_t *labels = new faiss::idx_t[k]; // Use faiss::idx_t for labels
		float *distances = new float[k];
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], _k, distances, labels);
			float min_dist = 1e30;
			int min_index = -1;
			for (int j = 0; j < _k; ++j)
			{
				faiss::idx_t neighbor = labels[j];
				float dist = distances[j];
				if (neighbor != i && dist < min_dist)
				{
					min_dist = min_dist;
					min_index = neighbor;
				}
			}
			clusters[i] = result[min_index];
		}

		delete[] dataset;
		delete[] sampleset;
		delete[] labels;
		delete[] distances;

		return clusters;
	}

};

#endif