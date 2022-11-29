from kite_text_source import kite_text, kite_history
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from collections import OrderedDict
import copy

tokenizer = TreebankWordTokenizer()

kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)

kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)

intro_total = len(intro_tokens)
history_total = len(history_tokens)
print('intro total:', intro_total)
print('history total:', history_total)

intro_tf = {}
history_tf = {}

intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total

history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total

print("Term Frequency of 'kite' in intro is: {:.4f}".format(intro_tf['kite']))
print("Term Frequency of 'kite' in history is: {:.4f}".format(history_tf['kite']))

# compare it with word 'and'
intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
print("\nCompare it with word 'and':")
print("Term Frequency of 'and' in intro is: {:.4f}".format(intro_tf['and']))
print("Term Frequency of 'and' in history is: {:.4f}".format(history_tf['and']))

num_docs_containing_and = 0
num_docs_containing_kite = 0
num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1
    if 'kite' in doc:
        num_docs_containing_kite += 1
    if 'china' in doc:
        num_docs_containing_china += 1

intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total

num_docs = 2
intro_idf = {}
history_idf = {}

intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and

intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite

intro_idf['china'] = num_docs / num_docs_containing_china
history_idf['china'] = num_docs / num_docs_containing_china

intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']

history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']

# t - term, d - document, D - corpus
# tf(t, d) = count(t) / count(d)
# idf(t, D) = log (number of documents / number of documents containing t)
# tfidf(t, d, D) = tf(t, d) * idf(t, D)
#
# log_tf = log(term_occurrences_in_doc) - log(num_terms_in_doc)
# log_log_idf = log(log(total_num_docs) - log(num_docs_containing_term))
# log_tf_idf = log_tf + log_idf

print('\n\n\n>>> RELEVANCE RANKING')
docs = [kite_intro, kite_history]

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

# print(document_tfidf_vectors)
print('cosine similarity')

from e2_compute_cosine_similarity import cosine_sim

print('len documents_tfidf_vectors:', len(document_tfidf_vectors))
print(cosine_sim(document_tfidf_vectors[0], document_tfidf_vectors[1]))

