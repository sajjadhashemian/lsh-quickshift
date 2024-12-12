// clang++ -std=c++17 -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfaiss -Wl,-rpath,$CONDA_PREFIX/lib -o quick_shift lsh-ms.cpp

#include <bits/stdc++.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Clustering.h>
using namespace std;

// srand(static_cast<unsigned>(1337));

class QuickShift
{
public:
	// QuickShift(float bandwidth, int k_neighbors) : h(bandwidth), k(k_neighbors) {}
	QuickShift(int k_neighbors) : k(k_neighbors) {}

	vector<int> fit(const vector<vector<float>> &data)
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

		// Compute density for each point
		vector<float> densities(n, 0.0);
		vector<faiss::idx_t> most_dense_neighbors(n, -1); // Use faiss::idx_t for labels

		// Find k-nearest neighbors for each point
		faiss::idx_t *labels = new faiss::idx_t[k]; // Use faiss::idx_t for labels
		float *distances = new float[k];
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);
			// Compute density using Gaussian kernel
			for (int j = 0; j < k; ++j)
			{
				float dist_sq = distances[j];
				densities[i] += exp(-dist_sq);
				// densities[i] += exp(-dist_sq + (rand() % 2));
				// densities[i] += exp(-dist_sq/(2*h*h));
			}
		}
		for (int i = 0; i < n; ++i)
		{
			index.search(1, &dataset[i * dim], k, distances, labels);
			// cout << "HERE " <<i<< ":";
			// Finding the most dense neighbor (excluding itself)
			float max_density = -1.0;
			faiss::idx_t most_dense_neighbor = -1;
			for (int j = 0; j < k; ++j)
			{
				faiss::idx_t neighbor = labels[j];
				// cout<<neighbor<<" ";
				if (neighbor != i && densities[neighbor] > max_density)
				{
					max_density = densities[neighbor];
					most_dense_neighbor = neighbor;
				}
			}
			// cout<<endl;
			if (densities[i] > max_density)
			{
				max_density = densities[i];
				most_dense_neighbor = i;
			}
			most_dense_neighbors[i] = most_dense_neighbor;
		}
		// Assign clusters based on the most dense neighbors
		vector<int> clusters(n, -1);
		int cluster_id = 0;
		for (int i = 0; i < n; ++i)
		{
			faiss::idx_t current = i;
			while (most_dense_neighbors[current] != current)
				// cerr << "FLAG!!!! " << i << " " << current << " " << densities[current]<<endl,
					current = most_dense_neighbors[current];
			if (clusters[current] == -1)
				clusters[current] = cluster_id++;
		}
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

	inline bool cointoss(double n)
	{
		int randomValue = rand();
		double threshold = (RAND_MAX / sqrt(n));
		return (bool)(randomValue < threshold);
	}

	vector<int> fast_fit(const vector<vector<float>> &data)
	{
		int n = data.size();
		int dim = data[0].size();
		int _k = sqrt(k) * 5;

		// sample sqrt(n) vector
		vector<vector<float>> sample;
		vector<int> index_sample;
		for (int i = 0; i < n; i++)
			if (cointoss(n))
			{
				sample.push_back(data[i]);
				index_sample.push_back(i);
			}
		int m = sample.size();
		cerr << sample.size() << " " << _k << endl;

		QuickShift qs(_k);
		vector<int> result = qs.fit(sample);
		vector<int> clusters(n, -1);
		for (int i = 0; i < index_sample.size(); ++i)
			clusters[index_sample[i]] = result[i];
		cerr << "HREE" << endl;

		// Prepare data for Faiss
		float *sampleset = new float[m * dim];
		for (int i = 0; i < m; ++i)
			for (int d = 0; d < dim; ++d)
				sampleset[i * dim + d] = data[i][d];
		float *dataset = new float[n * dim];
		for (int i = 0; i < n; ++i)
			for (int d = 0; d < dim; ++d)
				dataset[i * dim + d] = data[i][d];

		cerr << "HREE" << endl;

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

private:
	// float h; // Bandwidth for kernel density estimation
	int k; // Number of neighbors for k-NN search
};

// Generate random data with two clusters
vector<vector<float>> generate_data(int n, int dim)
{
	vector<vector<float>> data(2 * n, vector<float>(dim));
	default_random_engine generator;
	normal_distribution<float> dist0(0.0, 5.0);
	normal_distribution<float> dist1(7.0, 5.0);
	normal_distribution<float> dist2(90.0, 5.0);
	normal_distribution<float> dist3(60.0, 5.0);

	// for (int i = 0; i < n; ++i)
	// {
	// 	for (int d = 0; d < dim; ++d)
	// 	{
	// 		data[i][d] = dist0(generator);
	// 		data[n + i][d] = dist3(generator);
	// 	}
	// }
	for (int i = 0; i < n; ++i)
	{
		data[i][0] = dist0(generator);
		data[i][1] = dist1(generator);
		data[n + i][0] = dist2(generator);
		data[n + i][1] = dist3(generator);
	}

	return data;
}

int main()
{
	// Generate data
	int n = 10000; // Number of points per cluster
	int dim = 2;  // Dimensionality
	vector<vector<float>> data = generate_data(n, dim);
	// cout<<"Generated Data!"<<endl;
	for (int i = 0; i < 2 * n; i++)
	{
		for (int j = 0; j < dim; j++)
			cout << data[i][j] << " ";
		cout << endl;
	}

	QuickShift qs(sqrt(n)); // Bandwidth = 1.0, k = 50
	// auto x = qs.fit(data);
	auto x = qs.fast_fit(data);
	for (int y : x)
		cout << y << " ";
	cout << endl;
	return 0;
}