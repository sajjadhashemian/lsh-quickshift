#ifndef DSU_H
#define DSU_H
#include "common.h"

class DSU
{
private:
	int n;
	vector<int> parent;
	vector<int> size;
public:
	DSU(int num)
	{
		n = num;
		parent.resize(n);
		size.resize(n);
		for(int i=0;i<n;i++)
		{
			parent[i]=i;
			size[i]=0;
		}
	}
	~DSU()
	{
		parent.clear();
		size.clear();
	}

	int find_set(int v)
	{
		if (v == parent[v])
			return v;
		return parent[v] = find_set(parent[v]);
	}

	void union_sets(int a, int b)
	{
		a = find_set(a);
		b = find_set(b);
		if (a != b)
		{
			if (size[a] < size[b])
				swap(a, b);
			parent[b] = a;
			size[a] += size[b];
		}
	}
};

#endif