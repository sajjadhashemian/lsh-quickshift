// g++ -std=c++17 -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfaiss -Wl,-rpath,$CONDA_PREFIX/lib -o qs qs.cpp

#include "common.h"
#include "quickshift.h"

vector<int> lsh_quickshift(const vector<vector<float>> &data, int k, int n_bits = 128)
{
	QuickShift qs(k);
	return qs.fit(data);
}

vector<int> fast_lsh_quickshift(const vector<vector<float>> &data, int k, int _k, int n_bits = 128)
{
	QuickShift qs(k, _k);
	return qs.fast_fit(data);
}

int main()
{
	int n, dim;
	cin >> n >> dim;
	vector<vector<float>> data;
	for (int i = 0; i < n; i++)
		for(int d=0;d<dim;d++)
			cin>>data[i][d];

	int k, _k, n_bits;
	cin>>k;
	QuickShift qs(k);
	// auto x = qs.fit(data);
	auto x = qs.fast_fit(data);
	for (int y : x)
		cout << y << " ";
	cout << endl;
	return 0;
}