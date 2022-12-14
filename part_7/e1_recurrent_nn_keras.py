import glob
import os
import numpy as np
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN


word_vectors = get_data('wv', limit=200_000)


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


dataset = preprocess_data("../src/part_6/stanford_sent_analysis_dataset/aclImdb/train")
print("Because I don't have enough memory: I'll crop the dataset:")
dataset = dataset[:5000]
print('Len new dataset:', len(dataset))
vectorized_data, expected = tokenize_and_vectorize(dataset)
del dataset
del word_vectors

split_point = int(len(vectorized_data) * .8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]

x_test = vectorized_data[split_point:]
y_test = expected[split_point:]
del expected

print("\n\nTotal dataset parameters:")
print("Dataset size:", len(vectorized_data))
print("Training set size:", len(x_train))
print("Test set size:", len(x_test), "\n\n")
del vectorized_data

# RNN params:
maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 3
num_neurons = 50

# pad + trunkate the data
x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

# build the RNN
model = Sequential()
model.add(SimpleRNN(units=num_neurons,
                    return_sequences=True,
                    input_shape=(maxlen, embedding_dims)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("\n\nBuilding the RNN model:")
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# save model to file:
model_structure = model.to_json()
with open('../src/part_7/rnn_model_50n/rnn_model.json', 'w') as json_file:
    json_file.write(model_structure)
model.save_weights('../src/part_7/rnn_model_50n/rnn_weights.h5')
