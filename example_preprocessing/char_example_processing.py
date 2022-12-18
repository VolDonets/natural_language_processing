import numpy as np


def avg_symbol_len(data):
    total_len = 0
    for sample in data:
        total_len += len(sample[1])
    return total_len / len(data)


def clear_data(data):
    """Shift to lower case, replace unknowns with UNK, and listify"""
    new_data = []
    VALID = 'abcdefghijklmnopqrstuvwxyz0123456789"\'?!.,:; '
    for sample in data:
        new_sample = []
        for char in sample[1].lower():
            if char in VALID:
                new_sample.append(char)
            else:
                new_sample.append('UNK')
        new_data.append(new_sample)
    return new_data


def char_pad_trunc(data, maxlen=1500):
    """We truncated to maxlen or add in PAD tokens"""
    new_dataset = []
    for sample in data:
        if len(sample) > maxlen:
            new_data = sample[:maxlen]
        elif len(sample) < maxlen:
            pads = maxlen - len(sample)
            new_data = sample + ['PAD'] * pads
        else:
            new_data = sample
        new_dataset.append(new_data)
    return new_dataset


def create_dicts(data):
    """Mod from the Keras LSTM example"""
    chars = set()
    for sample in data:
        chars.update(set(sample))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char


def one_hot_encode(dataset, char_indices, maxlen=1500):
    """
    One-hot encode the tokens
    :param dataset: list of list tokens
    :param char_indices: dict of {key=character, value=index to use encoding vector}
    :param maxlen: int length of each sample
    :return: np array of shape (samples, tokens, encoding length)
    """

    X = np.zeros((len(dataset), maxlen, len(char_indices.keys())))
    for i, sentence in enumerate(dataset):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1

    return X
