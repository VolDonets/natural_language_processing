from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]

A = imread('../src/images/103041224_p0.jpg')
X = np.mean(A, -1)  # conv RGB to grayscale

# img = plt.imshow(X)
img = plt.imshow(256-X)   # to negative
img.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 20, 100):
    # Construct approximate image
    X_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure(j+1)
    j += 1
    # img = plt.imshow(256-X_approx)
    img = plt.imshow(X_approx)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r =' + str(r))
    plt.show()

# plot singular values
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sume')
plt.show()
