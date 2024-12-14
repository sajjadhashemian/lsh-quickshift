// g++ -std=c++17 -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfaiss -Wl,-rpath,$CONDA_PREFIX/lib -o kde kde.cpp

#include "common.h"
#include "kde.h"
using namespace std;


// Sample generator: Generate a dataset of random points
vector<vector<float>> generate_sample_data(int n, int d)
{
	vector<vector<float>> dataset(n, vector<float>(d));
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-1.0, 1.0);

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < d; ++j)
		{
			dataset[i][j] = dis(gen);
		}
	}

	return dataset;
}

vector<vector<float>> generate_data(int n, int dim)
{
	vector<vector<float>> data(2 * n, vector<float>(dim));
	default_random_engine generator;
	normal_distribution<float> dist0(0.0, 0.0);
	normal_distribution<float> dist3(60.0, 0.0);

	for (int i = 0; i < n; ++i)
	{
		for (int d = 0; d < dim; ++d)
		{
			data[i][d] = dist0(generator);
			data[n + i][d] = dist3(generator);
		}
	}

	return data;
}

vector<vector<float>> generate_query(int n, int dim)
{
	vector<vector<float>> data(2 * n, vector<float>(dim));
	default_random_engine generator;
	for (int i = 0; i < n; ++i)
	{
		for (int d = 0; d < dim; ++d)
		{
			normal_distribution<float> dist0(0.0, double(i));
			normal_distribution<float> dist3(60.0, double(i));
			data[i][d] = dist0(generator);
			data[n + i][d] = dist3(generator);
		}
	}

	return data;
}

int main()
{
	// Parameters
	int n = 100000;	  // Number of data points
	int d = 15;	  // Dimension of the data
	int n_bits = 128; // Number of bits for LSH
	int k = sqrt(n);		  // Number of nearest neighbors to retrieve

	// Generate sample data
	vector<vector<float>> dataset = generate_data(n, d);

	// Create KDE object
	KDE kde(d, n_bits);

	// Preprocess: Build the LSH index
	kde.fit(dataset);

	// Query: Compute the kernel density estimate
	auto query_point = generate_query(5, d);
	for(auto x:query_point)
	{
		auto estimate = kde.query(x, k);
		for(auto y:x)
			cout<<y<<", ";
		cout << "Kernel Density Estimate: " << estimate << endl;
	}


	return 0;
}