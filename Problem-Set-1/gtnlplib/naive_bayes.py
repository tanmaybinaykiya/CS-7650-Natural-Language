from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation
from collections import Counter
import numpy as np
from collections import defaultdict
import math

# deliverable 3.1
def get_corpus_counts(x, y, label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of Counters, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: corpus counts
    :rtype: defaultdict

    """

    final = Counter()

    for (index, yi) in enumerate(y):
        if label == yi:
            final = x[index] + final

    return defaultdict(float, final)


# deliverable 3.2
def estimate_pxy(x, y, label, smoothing, vocab):
    """
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, List of counters per document
    :param y: list of labels, List of strings per document
    :param label: desired label, String
    :param smoothing: additive smoothing amount, Float
    :param vocab: list of words in vocabulary, Counter of all words
    :returns: log probabilities per word
    :rtype: defaultdict

    """

    # phi_y_j = (smoothing + count(y, j)) / ( V x smoothing + sumn(j')(count(y, j') ) )
    # count(y, j) refers to the count of word j in documents with label y.

    # print("X:", x)
    # print("Y:", y)
    # print("label:", label)
    # print("smoothing:", smoothing)
    # print("vocab:", vocab)

    #####################################################################

    V = len(vocab)

    all_docs_with_label = [x[i] for i in range(len(y)) if y[i] == label]

    count_aggregator_for_label = Counter()

    for doc in all_docs_with_label:
        count_aggregator_for_label.update(doc)

    no_of_words_in_docs_with_label = len(list(count_aggregator_for_label.elements()))

    log_phi = defaultdict(float)

    denom = np.log(no_of_words_in_docs_with_label + ((V+1) * smoothing))
    for (word, _) in vocab:
        log_phi[word] = np.log(count_aggregator_for_label[word] + smoothing) - denom

    return log_phi


# deliverable 3.3
def estimate_nb(x, y, smoothing):
    """estimate a naive bayes model

    :param x: list of documents each represented by a counter of words
    :param y: list of labels
    :param smoothing: smoothing constant
    :param weights: a defaultdict of features and weights. features are tuples (label, base_feature).
    :rtype: defaultdict 

    """

    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)

    # print("X:", x)
    # print("Y:", y)
    # print("smoothing:", smoothing)

    unique_labels = list(set(y))

    vocab = defaultdict()

    for xi in x:
        vocab.update(xi)

    theta = []

    for label in unique_labels:
        theta.append(estimate_pxy(x, y, label, smoothing, vocab))

    return theta


# deliverable 3.4
def find_best_smoother(x_tr, y_tr, x_dv, y_dv, smoothers):
    """
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    """

    raise NotImplementedError
