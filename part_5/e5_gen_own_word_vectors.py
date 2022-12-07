# preprocessing step:
# 1. Break your documents into sentences
# 2. Break sentences into tokens
# So the training input should look similar to the following structure:

# >>> token_list = [
# >>>    ['to', 'provide', 'early', ...],
# >>>    ['essential', 'job', 'functions', ...],
# >>>    ['participate', ...]]
# >>> ]
token_list = []

# train your domain-specific word2vec model
from gensim.models.word2vec import Word2Vec

num_features = 300  # number of vector elements (dimensions) to represent the word vector
min_word_count = 3  # min number of word count to be considered in the Word2vec model
#                     If corpus's small - reduce, otherwise - increase
num_workers = 2     # Number of CPU cores used for the training.
#                     Possible to set num of cores dynamically.
#                     num_workers = multiprocessing.cpu_count()
window_size = 6     # Context window size
subsampling = 1e-3  # Subsampling rate for frequent terms

model = Word2Vec(token_list,
                 workers=num_workers,
                 size=num_features,
                 min_count=min_word_count,
                 window=window_size,
                 sample=subsampling)

# for memory saving, we can freeze the model and discard the unnecessary info:
model.init_sims(replace=True)

# saving the model:
model_name = 'my_domain_specific_word2vec_model'
model.save(model_name)

# for loading model:
model = Word2Vec.load(model_name)

# for the model testing you can use methods from 'e4_gensim_word2vec_module_usage.py'
print(model.most_similar('radiology'))

