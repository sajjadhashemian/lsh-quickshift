#ifndef COMMON_H
#define COMMON_H
#include <bits/stdc++.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Clustering.h>
#include <faiss/IndexLSH.h>
#include <faiss/index_io.h>

using namespace std;

bool __cointoss(double n)
{
	int randomValue = rand();
	double threshold = (RAND_MAX / sqrt(n));
	return (bool)(randomValue < threshold);
}
#endif