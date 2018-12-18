import os
import sys
import numpy as np
from collections import defaultdict, Counter
from gensim.models import KeyedVectors

vocab = defaultdict(float)

def clean_line(line):
    return line.replace('\n', '').replace('\u3000', '') \
                                 .replace('?', '') \
                                 .replace('!', '')


# Danger!
def clean_line_array(line_words):
    i = 0
    while i < len(line_words):
        if line_words[i] == '':
            del line_words[i]
        i += 1

        
def load_data_files(tag, dirname, word_count):
    """
    Load the data files and do some cleanup and pre-processing
    """
    filenames = os.listdir(dirname)

    data = []
    
    for fn in filenames:
        if fn.endswith("_processed.txt"):
            with open(os.path.join(dirname, fn), "r") as f:
                words = []

                for line in f:
                    line_words = clean_line(line).split(' ')
                    clean_line_array(line_words)
                    words += line_words

                for word in words:
                    vocab[word] += 1
                    word_count[word] += 1
                
                num_words = len(words)
                train = np.random.randint(0, 10)
                datum = {'category' : tag,
                         'words' : words,
                         'num_words' : num_words,
                         'train?' : train}
                data.append(datum)

    return data


def get_word_matrix(word_vecs, k=50):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_matrix = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1

    for word in word_vecs:
        W[i] = word_vecs      


def pad_all(data, word_count):
    """ 
    Pad all word arrays to max. length
    """
    max_len = max(d['num_words'] for d in data)

    # print('Max length is: {}'.format(max_len))
    
    padded = []

    for d in data:
        if d['num_words'] < max_len:
            pad_len = max_len - d['num_words']
            vocab['PAD'] += pad_len
            word_count['PAD'] += pad_len
            padded_words = d['words'] + ['PAD'] * pad_len
        else:
            padded_words = d['words']

        padded_datum = {'category' : d['category'],
                        'words' : padded_words,
                        'train?' : d['train?']}
        padded.append(padded_datum)

    return padded


def separate_data(padded):
    """
    Separate into training and testing sets
    """
    train = []
    test = []
    
    for d in padded:
        if d['train?'] < 7:
            train.append({'category' : d['category'],
                          'words' : d['words']})
        else:
            test.append({'category' : d['category'],
                         'words' : d['words']})

    return train, test


def produce_label_dict(labels):
    """
    Represent labels using one-hot vectors
    """
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    return label_dict


# Adapted from: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def build_vocab(word_count):
    vocabulary_inv = [word[0] for word in word_count.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv
            

def gen_xy_vectors(data, labels, vocabulary):
    label_dict = produce_label_dict(labels)
    x = []
    y = []
    
    for d in data:
        word_list = d['words']
        word_int_list = []
        for word in word_list:
            word_int_list.append(vocabulary[word])
        x.append(word_int_list)
        y.append(label_dict[d['category']])

    return np.array(x), np.array(y)


# Each directory is associated with a tag
def process_data(dir_pairs):
    data = []
    tags = []

    word_count = Counter()
    
    for tag, dirname in enumerate(dir_pairs):
        data += load_data_files(tag, dirname, word_count)
        tags.append(tag)

    data = pad_all(data, word_count)

    vocabulary, vocabulary_inv = build_vocab(word_count)
        
    x, y = gen_xy_vectors(data, tags, vocabulary)

    return x, y, vocabulary, vocabulary_inv, tags


