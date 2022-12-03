# LSA ~ Latent semantic analysis
# SVD ~ singular value decomposition
# LSA is based on SVD
# truncated singular value decomposition (truncated SVD) is used for LSA
# truncated SVD also called PCA (principal component analysis)
# LSA on natural language is equivalent to PCA on TF-IDF vectors.

import numpy as np
import pandas as pd

from nlpia.book.examples.ch04_catdog_lsa_3x6x16 import word_topic_vectors
from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

print(word_topic_vectors.T.round(1))

bow_svd, tfidf_svd = lsa_models()
print(prettify_tdm(**bow_svd))

tdm = bow_svd['tdm']
print(tdm)

U, s, Vt = np.linalg.svd(tdm)
print(pd.DataFrame(U, index=tdm.index).round(2))

print('\n\nS - singular values')
print(s.round(1))
S = np.zeros((len(U), len(Vt)))
np.fill_diagonal(S, s)
print(pd.DataFrame(S).round(1))

s_norm = s / (np.sqrt(np.sum(s * s) / len(s)))
s_norm = s_norm / np.sum(s_norm)
print('\ns_norm:', s_norm)

print('\n\nVt:')
print(pd.DataFrame(Vt).round(2))

err = []
for numdim in range(len(s), 0, -1):
    S[numdim-1, numdim-1] = 0
    reconstructed_tdm = U.dot(S).dot(Vt)
    err.append(np.sqrt(((reconstructed_tdm - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape)))
print('\n\nReconstruction Errors:', np.array(err).round(2))


