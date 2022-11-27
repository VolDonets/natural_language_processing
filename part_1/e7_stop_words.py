stop_words = ['a', 'an', 'the', 'on', 'of', 'off', 'this', 'is']
tokens = ['the', 'house', 'is', 'on', 'fire']
tokens_without_stopwords = [x for x in tokens if x not in stop_words]
print(tokens_without_stopwords, '\n')


# print list of nltk stop words
import nltk

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
print('Count NLTK stop words:', len(stop_words))
print(stop_words[:10], '\n')

# print sklearn stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

print('Count sklearn stop words:', len(sklearn_stop_words))
print(list(sklearn_stop_words)[:10])
