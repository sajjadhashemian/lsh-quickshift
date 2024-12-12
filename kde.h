#include "common.h"

class KDE
{
private:
	faiss::IndexLSH *index;		   // LSH index
	vector<vector<float>> dataset; // Dataset
	int n_bits;					   // Number of bits for LSH
	int d;						   // Dimension of the data
	double __sigma;		   // Gaussian kernel parameter

	// Gaussian kernel function
	double gaussian_kernel(const vector<float> &p, const vector<float> &q)
	{
		double dist_sq = 0.0;
		for (size_t i = 0; i < p.size(); ++i)
			dist_sq += (p[i] - q[i]);
		return exp(-dist_sq / __sigma);
	}

public:
	// Constructor
	KDE(int dimension, int bits, double s = 2) : d(dimension), n_bits(bits), __sigma(s), index(nullptr) {}

	// Destructor
	~KDE()
	{
		if (index)
			delete index;
	}

	// Preprocess: Build the LSH index
	void fit(const vector<vector<float>> &data)
	{
		dataset = data;
		int n = dataset.size();

		// Convert dataset to a flat array for FAISS
		float *data_array = new float[n * d];
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < d; ++j)
			{
				data_array[i * d + j] = dataset[i][j];
			}
		}

		// Create and train the LSH index
		index = new faiss::IndexLSH(d, n_bits);
		index->train(n, data_array);
		index->add(n, data_array);

		delete[] data_array;
	}

	// Query: Compute the kernel density estimate
	double query(const vector<float> &query_point, int k = 10)
	{
		if (!index)
		{
			cerr << "Error: LSH index not initialized. Call preprocess() first." << endl;
			return 0.0;
		}

		// Convert query point to a flat array
		float *query_data = new float[d];
		for (int i = 0; i < d; ++i)
			query_data[i] = query_point[i];

		// Search for nearest neighbors
		vector<faiss::idx_t> indices(k); // Use faiss::idx_t
		vector<int> __indices(k);
		vector<float> distances(k);
		index->search(1, query_data, k, distances.data(), indices.data());

		// Calculate the kernel density estimate
		double kde = 0.0;
		for (int i = 0; i < k; ++i)
		{
			int idx = indices[i];
			double weight = gaussian_kernel(dataset[idx], query_point);
			kde += weight;
			__indices[i]=idx;
		}
		kde /= dataset.size();

		delete[] query_data;
		return kde;
	}
};
