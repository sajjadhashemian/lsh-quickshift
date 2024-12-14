// g++ -std=c++17 -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfaiss -Wl,-rpath,$CONDA_PREFIX/lib -o qs qs.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For STL containers like vector
#include <pybind11/eigen.h>

#include "common.h"
#include "quickshift.h"

namespace py = pybind11;

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


PYBIND11_MODULE(qs, m)
{
	m.doc() = "pybind11 bindings for QuickShift";

	m.def("lsh_quickshift", [](const std::vector<std::vector<float>> &data, int k, int n_bits)
		  {
        QuickShift qs(k);
        return qs.fit(data); }, py::arg("data"), py::arg("k"), py::arg("n_bits") = 128, "Compute LSH QuickShift.");

	m.def("fast_lsh_quickshift", [](const std::vector<std::vector<float>> &data, int k, int _k, int n_bits)
		  {
        QuickShift qs(k, _k);
        return qs.fast_fit(data); }, py::arg("data"), py::arg("k"), py::arg("_k"), py::arg("n_bits") = 128, "Compute Fast LSH QuickShift.");
}