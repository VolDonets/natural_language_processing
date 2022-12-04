import pandas as pd
import numpy as np
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


pd.options.display.width = 120

sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!' * j) for (i, j) in zip(range(len(sms)), sms.spam)]

sms.index = index
print(sms.head(6))

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
print('\nLen TF-IDF vocabulary:', len(tfidf.vocabulary_))

tfidf_docs = pd.DataFrame(tfidf_docs)
# center vectorized documents (BOW vectors) by subtracting the mean
tfidf_docs = tfidf_docs - tfidf_docs.mean()
print('TF-IDF shape:', tfidf_docs.shape)
print("sms.spam.sum()", sms.spam.sum())


pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
print(pca_topic_vectors.round(3).head(6))

# print(tfidf.vocabulary_)
column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys())))
# print(terms)

# get weights from PCA for words in vocabulary
weights = pd.DataFrame(pca.components_, columns=terms, index=['topic{}'.format(i) for i in range(16)])
pd.options.display.max_columns = 0
print(weights.head(4).round(3))

print('\n\nShow weights for each topic for words in example')
pd.options.display.max_columns = 12
deals = weights['! ;) :) half off free crazy deal only $ 80 %'.split()].round(3) * 100
print(deals)
print('\ndeals.T.sum()\n', deals.T.sum())

print('\n\nTESTING and comparing TruncatedSVD Vs PCA')
svd = TruncatedSVD(n_components=16, n_iter=100)
svd_topic_vectors = svd.fit_transform(tfidf_docs.values)
svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns, index=index)
print(svd_topic_vectors.round(3).head(6))

# Normalize each topic vector by its length (L2-norm) allows you to compute the cosine
# distance with a dot product
svd_topic_vectors = (svd_topic_vectors.T / np.linalg.norm(svd_topic_vectors, axis=1)).T
print('\n\nCheck LSA for spam classification')
print(svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1))
