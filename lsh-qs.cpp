#include "common.h"
#include "quickshift.h"

// Generate random data with two clusters
vector<vector<float>> generate_data(int n, int dim)
{
	vector<vector<float>> data(2 * n, vector<float>(dim));
	default_random_engine generator;
	normal_distribution<float> dist0(0.0, 5.0);
	normal_distribution<float> dist1(7.0, 5.0);
	normal_distribution<float> dist2(90.0, 5.0);
	normal_distribution<float> dist3(60.0, 5.0);

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
	int n = 1000; // Number of points per cluster
	int dim = 2;   // Dimensionality
	vector<vector<float>> data = generate_data(n, dim);
	// cout<<"Generated Data!"<<endl;
	for (int i = 0; i < 2 * n; i++)
	{
		for (int j = 0; j < dim; j++)
			cout << data[i][j] << " ";
		cout << endl;
	}

	// QuickShift qs(sqrt(n));
	QuickShift qs(sqrt(n));
	auto x = qs.fit(data, 32);
	// auto x = qs.fast_fit(data);
	for (int y : x)
		cout << y << " ";
	cout << endl;
	return 0;
}