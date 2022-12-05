# LDA ~ Linear discriminant analysis
# LDA works similarly to LSA except it requires classification labels
# LSA maximizes the separation (variance) between vectors in the new space
# LDA maximizes the distance between the centroids of the vectors within each class

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from nlpia.data.loaders import get_data
from nltk.tokenize.casual import casual_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  train_test_split
from sklearn.decomposition import PCA

import pandas as pd


sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!' * j) for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()

lda = LDA(n_components=1)
lda = lda.fit(tfidf_docs, sms.spam)
sms['lda_spaminess'] = lda.predict(tfidf_docs)
print('Error:', ((sms.spam - sms.lda_spaminess) ** 2.).sum() ** .5)
print('Count right answers:', (sms.spam == sms.lda_spaminess).sum())
print('Count sms:', len(sms))
print('By results, probably, we got overfitting\n\n')

lda = LDA(n_components=1)
scores = cross_val_score(lda, tfidf_docs, sms.spam, cv=5)
print("Let's do some validaton:")
print('Accuracy: {:.2f} (+/-{:.2f})\n\n'.format(scores.mean(), scores.std() * 2))

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs, sms.spam, test_size=0.33, random_state=271828)
lda = LDA(n_components=1)
lda.fit(X_train, y_train)
print('Print accuracy with train/test splitting:')
print('Train acc:', lda.score(X_train, y_train).round(3))
print('Test acc:', lda.score(X_test, y_test).round(3), '\n\n')

pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)

X_train, X_test, y_train, y_test = train_test_split(pca_topic_vectors.values, sms.spam, test_size=0.3, random_state=271828)
lda = LDA(n_components=1)
lda.fit(X_train, y_train)
print('Results with using PCA topic vectors:')
print('Train acc:', lda.score(X_train, y_train).round(3))
print('Test acc:', lda.score(X_test, y_test).round(3), '\n\n')

lda = LDA(n_components=1)
scores = cross_val_score(lda, pca_topic_vectors, sms.spam, cv=10)
print("Let's do some validaton with PCA topic vectors:")
print('Accuracy: {:.2f} (+/-{:.2f})\n\n'.format(scores.mean(), scores.std() * 2))

