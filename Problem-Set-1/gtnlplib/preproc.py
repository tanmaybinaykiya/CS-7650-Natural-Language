from collections import Counter

import pandas as pd
import numpy as np


# deliverable 1.1
def bag_of_words(text):
    """
    Count the number of word occurrences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    """
    
    return Counter(text.split())


# deliverable 1.2
def aggregate_counts_slow(bags_of_words):
    """
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    """

    counts = Counter()
    for bag in bags_of_words:
        counts.update(bag)
    return counts


# deliverable 1.2
def aggregate_counts(bags_of_words):
    """
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    """

    i = 0
    counts = Counter()
    for bag in bags_of_words:
        counts.update(bag)
    return counts


# deliverable 1.3
def compute_oov(bow1, bow2):
    """
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    """

    return set(bow1) - set(bow1 & bow2)


# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    """
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    """

    updated_training_counts = Counter(Counter({k: c for k, c in training_counts.items() if c >= min_counts}))
    new_target_data = list(map(lambda target: target & updated_training_counts, target_data))
    vocab = set(updated_training_counts.items())
    return new_target_data, vocab


# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    """
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    """
    vocab = sorted(vocab)
    V = len(vocab)
    N = len(bags_of_words)
    print("V:", V, "N:", N)

    np_bag_of_words = np.zeros(shape=(N, V))

    for i in range(N):
        for (j, _type) in enumerate(vocab):
            np_bag_of_words[i, j] = bags_of_words[i][_type[0]]

    return np_bag_of_words


# helper code
def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]


def oov_rate(bow1, bow2):
    return len(compute_oov(bow1, bow2)) / len(bow1.keys())
