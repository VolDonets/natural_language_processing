# LDA ~ Linear Discriminant analysis

import pandas as pd
from nlpia.data.loaders import get_data

pd.options.display.width = 120

sms = get_data('sms-spam')
index = ['sms{}{}'.format(i, '!' * j) for (i, j) in zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)

print('len(sms):', len(sms))
print('sms.spam.sum():', sms.spam.sum())
print(sms.head(10))

# now let's do our tokenization and TF-IDF vec-transformation on all these SMS messages:
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
print('tfidf_docs.shape:', tfidf_docs.shape)
print('sms.spam.sum():', sms.spam.sum())

# using LDA
# sklearn.discriminant_analysis.LinearDiscriminantAnalysis
# but it's too simply, so we do it directly:
mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)
print('Spam Centroid')
print(spam_centroid.round(2))
print('Ham Centroid')
print(ham_centroid.round(2))

# The dot product computes the 'shadow' or projection of each vector on the line
# between the centroids.
spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
print('Spamminess Score')
print(spamminess_score.round(2))

# get score 0 - 1, like a probability
from sklearn.preprocessing import MinMaxScaler

sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1, 1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)
print(sms['spam lda_predict lda_score'.split()].round(2).head(20))

# with threshold as 50%
print('Prob to predict spam:', (1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3))

# show confusion matrix
from pugnlp.stats import Confusion
print(Confusion(sms['spam lda_predict'.split()]))

