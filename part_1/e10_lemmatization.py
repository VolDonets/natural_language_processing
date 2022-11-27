import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
print('better ->', lemmatizer.lemmatize("better"))
print('better (with pos=a) ->', lemmatizer.lemmatize("better", pos='a'))
print('good (with pos=a) ->', lemmatizer.lemmatize("good", pos='a'))
print('goods (with pos=a) ->', lemmatizer.lemmatize("goods", pos='a'))
print('goods (with pos=n) ->', lemmatizer.lemmatize("goods", pos='n'))
print('goodness (with pos=n) ->', lemmatizer.lemmatize("goodness", pos='n'))
print('best (with pos=a) ->', lemmatizer.lemmatize("best", pos='a'))
