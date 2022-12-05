# LDiA ~ Latent Dirichlet allocation

import numpy as np
import pandas as pd
from nlpia.data.loaders import get_data
from nltk.tokenize.casual import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!' * j) for (i, j) in zip(range(len(sms)), sms.spam)]
sms.index = index

total_corpus_len = 0
for document_text in sms.text:
    total_corpus_len += len(casual_tokenize(document_text))
mean_document_len = total_corpus_len / len(sms)
print('Mean document len =', round(mean_document_len, 2))

# or in one line
# mean_document_len = sum([len(casual_tokenize(t)) for t in sms.text]) * 1. / len(sms.text)

# Apply LDiA on sms-spam
# LDiA requires BOW vectors instead normalized TF-IDF vectors
np.random.seed(42)

counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(), counter.vocabulary_.keys())))
bow_docs.columns = terms
print('Some checks:')
print(sms.loc['sms0'].text)
print(bow_docs.loc['sms0'][bow_docs.loc['sms0'] > 0].head())

ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)
print('\n\nldia.components_.shape =', ldia.components_.shape)

pd.set_option('display.width', 75)
columns = ['topic{}'.format(i) for i in range(ldia.n_components)]
components = pd.DataFrame(ldia.components_.T, index=terms, columns=columns)
print(components.round(2).head(3))
print(components.topic3.sort_values(ascending=False)[:10])

ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index=index, columns=columns)
print('\nLDiA 16 topic-vectors:')
print(ldia16_topic_vectors.round(2).head())

X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.5, random_state=271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
print('Test Accuracy on LDiA:', round(float(lda.score(X_test, y_test)), 2))

# # !! TIP for iteration through all the combinations (pairs or triplets) of a set
# # of objects, you can use the built-in Python product() function:
# from itertools import product
# all_pairs = [(word1, word2) for (word1, word2) in product(word_list, word_list) if not word1 == word2]

# compare LDiA model with a model based on TF-IDF vectors:
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = tfidf_docs - tfidf_docs.mean(axis=0)

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs, sms.spam.values, test_size=0.5, random_state=271828)

lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
print('\n\nTrain accuracy on TF-IDF:', round(float(lda.score(X_train, y_train)), 3))
print('Test accuracy on TF-IDF:', round(float(lda.score(X_test, y_test)), 3))

