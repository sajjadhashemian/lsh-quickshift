from cpp import qs

data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
k = 2
n_bits = 128

result = qs.lsh_quickshift(data, k, n_bits)
print(result)