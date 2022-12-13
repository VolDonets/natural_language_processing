import numpy as np
from keras.models import model_from_json
from nlpia.loaders import get_data
from nltk.tokenize import TreebankWordTokenizer


with open("../src/part_6/cnn_models/cnn_model.json", 'r') as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights('../src/part_6/cnn_models/cnn_weights.h5')

sample_1 = "I hate that the dismal weather had me down for so long, " \
           "when will it break! Ugh, when does happiness return? The sun is blinding " \
           "and the puffy clouds are too thin. I can't wait for the weekend."

word_vectors = get_data('w2v', limit=200_000)


def tokenize_and_vectorize(dataset_):
    tokenizer_ = TreebankWordTokenizer()
    vectorized_data_ = []
    expected_ = []
    for sample_ in dataset_:
        tokens_ = tokenizer_.tokenize(sample_[1])
        sample_vecs_ = []
        for token_ in tokens_:
            try:
                sample_vecs_.append(word_vectors[token_])
            except KeyError:
                pass
        vectorized_data_.append(sample_vecs_)
        expected_.append(sample_[0])
    return vectorized_data_, expected_


def pad_trunc(data_, maxlen_):
    new_data_ = []
    zero_vector_ = [0.0 for _ in range(len(data_[0][0]))]
    for sample_ in data_:
        if len(sample_) > maxlen_:
            temp_ = sample_[:maxlen_]
        elif len(sample_) < maxlen_:
            temp_ = sample_
            additional_elems_ = maxlen_ - len(sample_)
            for _ in range(additional_elems_):
                temp_.append(zero_vector_)
        else:
            temp_ = sample_
        new_data_.append(temp_)
    return new_data_


maxlen = 400
embedding_dims = 300
dataset = [[1, sample_1]]
vec_list, expected = tokenize_and_vectorize(dataset)
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
prediction = model.predict(test_vec)
print('Positive statement: 1')
print('Negative statement: 0')
print('Prediction for negative sentence:', prediction)
