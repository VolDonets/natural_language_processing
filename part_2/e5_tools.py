from sklearn.feature_extraction.text import TfidfVectorizer
from kite_text_source import kite_text, kite_history

docs = list()
docs.append("The faster Harry got to the store, the faster and faster Harry would get home.")
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")
corpus = docs

vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)

# method '.todense()' converts a sparse matrix back into a regular numpy matrix
# (filling in the gaps with zeros) for your viewing pleasure.
print(model.todense().round(2))

