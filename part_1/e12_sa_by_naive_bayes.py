from nlpia.data.loaders import get_data
import pandas as pd
from nltk.tokenize import casual_tokenize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB


movies = get_data('hutto_movies')
print(movies.head().round(2), '\n')
print(movies.describe().round(2), '\n')

pd.set_option('display.width', 75)
bags_of_words = []

# without case rorm, stemming and lemmatization
for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))

# with case normalization:
# for text in movies.text:
#     bags_of_words.append(Counter(casual_tokenize(text.lower())))

# with stemming
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()
# for text in movies.text:
#     text_stemmed = ' '.join([stemmer.stem(w).strip("'") for w in text.split()])
#     bags_of_words.append(Counter(casual_tokenize(text_stemmed)))

# with lemmatization
# import nltk
# nltk.download('onw-1.4')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# for text in movies.text:
#     text_lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
#     bags_of_words.append(Counter(casual_tokenize(text_lemmatized)))

df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)
print("df_bows.shape:", df_bows.shape, '\n')
print(df_bows.head(), '\n')
print(df_bows.head()[list(bags_of_words[0].keys())])

nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)


movies['predicted_sentiment'] = nb.predict_proba(df_bows)[:, 1] * 8 - 4
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
print('Error mean:', movies.error.mean().round(1), '\n')

movies['sentiment_ispositive'] = (movies.sentiment > 0).astype(int)
movies['predicted_ispositive'] = (movies.predicted_sentiment > 0).astype(int)
print(movies["sentiment predicted_sentiment sentiment_ispositive predicted_ispositive".split()].head(8), '\n')
print((movies.predicted_ispositive == movies.sentiment_ispositive).sum() / len(movies))

# results without case norm, lemmatization and stemming
# df_bows: 20756
# probability for is_positive = 93.44%

# results with case norm
# df_bows: 18541
# probability for is_positive = 92.64%

# results with stemming
# df_bows: 16145
# probability for is_positive = 91.75%

# results with lemmatization
# df_bows: 20021
# probability for is_positive = 93.16%

print("\n\n\n>>>>> TESTING TRAINED ")
products = get_data('hutto_products')
bags_of_words = []
for text in products.text:
    bags_of_words.append(Counter(casual_tokenize(text)))

df_product_bows = pd.DataFrame.from_records((bags_of_words))
df_product_bows = df_product_bows.fillna(0).astype(int)
df_all_bows = df_bows.append(df_product_bows)
print(df_all_bows.columns)

df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]
df_product_bows = df_product_bows.fillna(0).astype(int)
print('df_product_bows.shape', df_product_bows.shape)
print('df_bows.shape:', df_bows.shape, '\n')

products['ispos'] = (products.sentiment > 0).astype(int)
products['predicted_ispositive'] = nb.predict(df_product_bows.values).astype(int)
print(products.head())
print('Product prediction:', (products.predicted_ispositive == products.ispos).sum()/len(products))

# and for the testing trained model on the test product dataset
# the model achieved 55.72% accuracy
