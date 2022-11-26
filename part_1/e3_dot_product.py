import numpy as np


v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])

print(v1.dot(v2))
print((v1 * v2).sum())

# not efficient way to do this
print(sum([x1 * x2 for x1, x2 in zip(v1, v2)]))

# !!! TIP
# for vectors in matrix can be used 'matrix product'
# np.matmul()

print(v1.reshape(-1, 1).T @ v2.reshape(-1, 1))
