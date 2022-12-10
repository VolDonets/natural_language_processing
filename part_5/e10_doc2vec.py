import multiprocessing
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess


num_cores = multiprocessing.cpu_count()
print("Count cores:", num_cores)

corpus = ["This is the first document ...",
          "This is the second document ...",
          "Another one."]
training_corpus = []
for i, text in enumerate(corpus):
    # gensim provides a data structure to annotate documents with string of integer tags
    # for catgory labels, keywords, ...
    tagged_doc = TaggedDocument(simple_preprocess(text), [i])
    training_corpus.append(tagged_doc)

# instantiate the Doc2vec object with your window size of 10 words and 100-D word and document vectors
model = Doc2Vec(size=100, min_count=2, workers=num_cores, iter=10)
# compile model before training
model.build_vocab(training_corpus)
# start training for 10 epochs
model.train(training_corpus, total_examples=model.corpus_count, epochs=model.iter)

# !!TIP for low RAM amount:
# use preallocated numpy array instead of Python list
# >>> training_corpus = np.empty(len(corpus), dtype=object)

# Infer document vectors for new, unseen document buy calling infer_vector
# on the instantiated and trained model:
model.infer_vector(simple_preprocess("This is a completely unseen document"), step=10)
