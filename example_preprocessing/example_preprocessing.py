import glob
import os
import numpy as np
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data


def tokenize_and_vectorize(dataset_, word_vectors_=None, w2v_limit=100_000):
    if word_vectors_ is None:
        word_vectors_ = get_data('w2v', limit=w2v_limit)
    tokenizer_ = TreebankWordTokenizer()
    vectorized_data_ = []
    expected_ = []
    for sample_ in dataset_:
        tokens_ = tokenizer_.tokenize(sample_[1])
        sample_vecs_ = []
        for token_ in tokens_:
            try:
                sample_vecs_.append(word_vectors_[token_])
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


def preprocess_data(filepath_):
    positive_path_ = os.path.join(filepath_, 'pos')
    negative_path_ = os.path.join(filepath_, 'neg')
    pos_label_ = 1
    neg_label_ = 0
    dataset_ = []

    for filename_ in glob.glob(os.path.join(positive_path_, '*.txt')):
        try:
            with open(filename_, 'r') as f_:
                dataset_.append((pos_label_, f_.read()))
        except UnicodeDecodeError:
            pass

    for filename_ in glob.glob(os.path.join(negative_path_, '*.txt')):
        try:
            with open(filename_, 'r') as f_:
                dataset_.append((neg_label_, f_.read()))
        except UnicodeDecodeError:
            pass

    shuffle(dataset_)
    return dataset_


def get_prepared_test_train_data(length_lim=-1, w2v_limit=-1, testing_split=.8, maxlen=400, embedding_dims=300):
    word_vectors = get_data('w2v') if w2v_limit < 0 else get_data('w2v', limit=w2v_limit)

    dataset = preprocess_data("../src/part_6/stanford_sent_analysis_dataset/aclImdb/test")
    print("Dataset length limit NOT applied" if length_lim < 0 else "Dataset length limit APPLIED")
    dataset = dataset if length_lim < 0 else dataset[:length_lim]
    print("Current size of dataset: ", len(dataset))

    vectorized_data, expected = tokenize_and_vectorize(dataset, word_vectors)
    del dataset, word_vectors

    split_point = int(len(vectorized_data) * testing_split)
    x_train = vectorized_data[:split_point]
    y_train = expected[:split_point]
    x_test = vectorized_data[split_point:]
    y_test = expected[split_point:]

    del vectorized_data, expected
    print("Count train samples:", len(x_train))
    print("Count test samples:", len(x_test))

    x_train = pad_trunc(x_train, maxlen)
    x_test = pad_trunc(x_test, maxlen)
    x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    y_train = np.array(y_train)
    x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def preprocess_to_nn(sample: str, expected_class: int = 1, maxlen=400,
                     word_vectors=None, w2v_limit=-1, embedding_dims=300):
    if word_vectors is None:
        word_vectors = get_data('w2v') if w2v_limit < 0 else get_data('w2v', limit=w2v_limit)

    dataset = [(expected_class, sample)]
    vec_list, expected = tokenize_and_vectorize(dataset, word_vectors)
    test_vec_list = pad_trunc(vec_list, maxlen)
    test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))

    return test_vec
