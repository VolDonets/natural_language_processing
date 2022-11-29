from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
import nltk
from collections import OrderedDict
import copy


nltk.download('stopwords', quiet=True)


sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)

bag_of_words = Counter(tokens)
print(bag_of_words)
print('Most common words:', bag_of_words.most_common(4))

times_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = times_harry_appears / num_unique_words
# tf ~ term frequency
print('TF("harry") =', round(tf, 4))

import kite_text_source
kite_text = kite_text_source.kite_text

tokens = tokenizer.tokenize(kite_text.lower())
tokens_count = Counter(tokens)
print('\n\n\n>>> Kite text tokens <<<')
print(tokens_count)

stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_count = Counter(tokens)
print('>>> Kite text after removing stopwords')
print(kite_count)


print("\n\n\n>>> Vectorizing")
document_vector = []
doc_length = len(tokens)
for key, value in kite_count.most_common():
    document_vector.append(value / doc_length)
print("Document vector:", document_vector)


docs = list()
docs.append("The faster Harry got to the store, the faster and faster Harry would get home.")
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
print("Len doc_tokens[0]:", len(doc_tokens[0]))

all_doc_tokens = sum(doc_tokens, [])
print("Len all_doc_tokens:", len(all_doc_tokens))

lexicon = sorted((set(all_doc_tokens)))
print("Len lexicon:", len(lexicon))
print(lexicon)

zero_vector = OrderedDict((token, 0) for token in lexicon)
print("Zero vector:", zero_vector)

doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)

print('\n\n\n>>> RELEVANCE RANKING')
document_tfidf_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)

    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc in docs:
            if key in _doc:
                docs_containing_key += 1
            tf = value / len(lexicon)
            if docs_containing_key:
                idf = len(docs) / docs_containing_key
            else:
                idf = 0
            vec[key] = tf * idf
    document_tfidf_vectors.append(vec)

print(document_tfidf_vectors)
print('cosine similarity')

from e2_compute_cosine_similarity import cosine_sim

print(cosine_sim(document_tfidf_vectors[0], document_tfidf_vectors[1]))
print(cosine_sim(document_tfidf_vectors[0], document_tfidf_vectors[2]))
print(cosine_sim(document_tfidf_vectors[1], document_tfidf_vectors[2]))

query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vector)

tokens = tokenizer.tokenize(query.lower())
token_counts = Counter(tokens)

for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in docs:
        if key in _doc.lower():
            docs_containing_key += 1
        if docs_containing_key == 0:
            continue
        tf = value / len(tokens)
        idf = len(docs) / docs_containing_key
        query_vec[key] = tf * idf

print('cosine sim to query & 0 =', cosine_sim(query_vec, document_tfidf_vectors[0]))
print('cosine sim to query & 1 =', cosine_sim(query_vec, document_tfidf_vectors[1]))
print('cosine sim to query & 2 =', cosine_sim(query_vec, document_tfidf_vectors[2]))

