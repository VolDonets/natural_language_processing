import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D

import glob
import os
from random import shuffle

from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data


word_vectors = get_data('w2v', limit=200_000)


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


def tokenize_and_vectorize(dataset_):
    tokenizer_ = TreebankWordTokenizer
    vectorized_data_ = []
    expected_ = []
    for sample_ in dataset_:
        tokens_ = tokenizer_.tokenize(tokenizer_, sample_[1])
        sample_vecs_ = []
        for token_ in tokens_:
            try:
                sample_vecs_.append(word_vectors[token_])
            except KeyError:
                pass
        vectorized_data_.append(sample_vecs_)
        expected_.append(sample_[0])
    return vectorized_data_, expected_


dataset = preprocess_data("../src/part_6/stanford_sent_analysis_dataset/aclImdb/test")
print('Result of loading a training dataset:')
print(dataset[0], '\n\n')
print('Len dataset:', len(dataset))
print("Because I don't have enough memory: I'll crop the dataset:")
dataset = dataset[:5000]
print('Len new dataset:', len(dataset))

vectorized_data, expected = tokenize_and_vectorize(dataset)
del dataset

split_point = int(len(vectorized_data) * .8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

del vectorized_data
del expected
del word_vectors

print("Count train samples:", len(x_train))
print("Count test samples:", len(x_test))

# CNN parameters:
maxlen = 400
batch_size = 32         # how many samples to show the net before
#                         backpropagation the error and updating the weights
embedding_dims = 300    # Length of the token vectors to create for passing into the convnet
filters = 250           # Number of filters for train
kernel_size = 3         # The width of the filters (filter matrix = embedding_dims x kernel_size
hidden_dims = 250       # Number of neurons in the plain feedforward net at the end of the chain
epochs = 10             # Number of times for passing the entire training dataset through the net


# 'keras' has a preprocessing helper method, 'pad_sequences', that in theory could be used to pad
# your input data, but it works only with sequences of scalars, not sequences of vectors.
# So let's create own method for padding sequences of vectors:
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


x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

print('\n\nBuild model ...')
model = Sequential()

model.add(Conv1D(
    filters=filters,
    kernel_size=kernel_size,
    padding='valid',
    activation='relu',
    strides=1,
    input_shape=(maxlen, embedding_dims)
))

# GlobalMaxPooping1D applies pooling to all feature map, not to a specific window
model.add(GlobalMaxPooling1D())

model.add(Dense(units=hidden_dims))
model.add(Dropout(rate=0.2))
model.add(Activation(activation='relu'))

model.add(Dense(units=1,
                activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# save model to file:
model_structure = model.to_json()
with open('../src/part_6/cnn_models/cnn_model.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('../src/part_6/cnn_models/cnn_weights.h5')
