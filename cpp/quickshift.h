#ifndef QUICKSHIFT_H
#define QUICKSHIFT_H

#include "common.h"
#include "kde.h"
#include "dsu.h"

class QuickShift
{
private:
	int k;		// Number of neighbors for k-NN search
	int n_bits; // Number of bits for LSH
	int dim;	// Dimension of the data
	int n;		// Number of the data
	int _k;

public:
	QuickShift(int k_neighbors, int _k_neighbors = 30) : k(k_neighbors), _k(_k_neighbors) {}

	vector<int> fit(const vector<vector<float>> &data, int n_bits = 128)
	{
		n = data.size();
		dim = data[0].size();

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
		DSU dsu(n);

		// Find k-nearest neighbors for each point
		faiss::idx_t *labels = new faiss::idx_t[k];
		float *distances = new float[k];
		float mn = 1000;
		float mx = -1;
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);

			// Compute density using KDE estimator
			vector<float> query_point(dim);
			for (int d = 0; d < dim; ++d)
				query_point[d] = data[i][d];
			density[i] = kde.query(query_point, k * 2);
		}

		// Assign each point to the nearest neighbor with higher density
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);

			// Find the neighbor with highest density
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
			dsu.union_sets(i, heaviest_neighbor);
		}

		// Assign clusters based on the parent structure
		vector<int> clusters(n, -1);
		int cluster_id = 0;
		map<int, pair<int, double>> mp;
		for (int i = 0; i < n; ++i)
		{
			int parent = dsu.find_set(i);
			if (clusters[parent] == -1)
			{
				mp[cluster_id]={-1,0};
				clusters[parent] = cluster_id++;
			}
		}

		// Propagate cluster assignments
		for (int i = 0; i < n; ++i)
		{
			if (clusters[i] == -1)
				clusters[i]=clusters[dsu.find_set(i)];
			pair<int, double> x=mp[clusters[i]];
			if(density[i]>x.second)
				mp[clusters[i]]={i, density[i]};
		}
		// for(int i=0;i<cluster_id;i++)
		// {
		// 	int idx = mp[i].first;
		// 	for(int d=0;d<dim;d++)
		// 		// cerr<<data[idx][j]<<" ",
		// 		cout<<data[idx][d]<<" ";
		// 	// cerr<<endl;
		// }
		// cout<<endl;

		delete[] dataset;
		delete[] labels;
		delete[] distances;

		return clusters;
	}

	vector<int> fast_fit(const vector<vector<float>> &data, int n_bits = 128)
	{
		int n = data.size();
		int dim = data[0].size();

		// sample ~sqrt(n) vector
		vector<vector<float>> sample;
		vector<int> index_sample;
		set<int> sample_index_set;
		for (int i = 0; i < n; i++)
			if (__cointoss(n))
			{
				sample.push_back(data[i]);
				index_sample.push_back(i);
				sample_index_set.insert(i);
			}
		
		// Adding ~sqrt(n) most dense data to sample for initial clustering
		KDE kde(dim, n_bits);
		kde.fit(sample);
		vector<pair<double, int>> density_sorted(n);
		vector<double> density(n);
		for (int i = 0; i < n; i++)
		{
			double rho = kde.query(data[i], k);
			density_sorted[i]={rho, i};
			density[i]=rho;
			// cerr<< rho <<endl;
		}
		sort(density_sorted.begin(), density_sorted.end(), greater<pair<double, int>>());

		int upper = sqrt(n)*log(n);
		upper = min(upper, n/10);

		for (int i = 0; i < n && sample_index_set.size()<upper; i++)
		{
			pair<double, int> x = density_sorted[i];
			int idx = x.second;
			if (sample_index_set.find(idx) == sample_index_set.end())
			{
				sample.push_back(data[idx]);
				index_sample.push_back(idx);
				sample_index_set.insert(idx);
			}
		}
		int m = sample.size();
		// int _k = log(m)*log(m);
		// cerr<< m << " " << _k <<endl;

		// Cluster sampled data
		QuickShift qs(_k);
		vector<int> result = qs.fit(sample);
		vector<int> clusters(n, -1);
		for (int i = 0; i < m; ++i)
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
		faiss::IndexFlatL2 index(dim);
		index.add(m, sampleset);

		// Find k-nearest neighbors for each point to assigning clusters
		faiss::idx_t *labels = new faiss::idx_t[_k];
		float *distances = new float[_k];
		for (int i = 0; i < n; ++i)
		{
			if (clusters[i] == -1)
			{
				index.search(1, &dataset[i * dim], _k, distances, labels);
				float min_dist = 1e30;
				float max_dens = 0;
				int min_index = -1;
				for (int j = 0; j < _k; ++j)
				{
					int neighbor = labels[j];
					float dist = distances[j];
					double rho = density[neighbor];
					if (neighbor != i && max_dens < rho)
					{
						min_dist = min_dist;
						max_dens = rho;
						min_index = neighbor;
					}
				}
				clusters[i] = result[min_index];
			}
		}

		delete[] dataset;
		delete[] sampleset;
		delete[] labels;
		delete[] distances;

		return clusters;
	}
};

#endif