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

    V = len(vocab)

    all_docs_with_label = [x[i] for i in range(len(y)) if y[i] == label]

    count_aggregator_for_label = Counter()

    for doc in all_docs_with_label:
        count_aggregator_for_label.update(doc)

    no_of_words_in_docs_with_label = len(list(count_aggregator_for_label.elements()))

    log_phi = defaultdict(float)

    denom = np.log(no_of_words_in_docs_with_label + (V * smoothing))
    for (word, _) in vocab:
        log_phi[word] = np.log(count_aggregator_for_label[word] + smoothing) - denom

    return log_phi


def estimate_pxy_2(x, y, label, smoothing, vocab):
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

    V = len(vocab)

    all_docs_with_label = [x[i] for i in range(len(y)) if y[i] == label]

    count_aggregator_for_label = Counter()

    for doc in all_docs_with_label:
        count_aggregator_for_label.update(doc)

    no_of_words_in_docs_with_label = len(list(count_aggregator_for_label.elements()))

    log_phi = defaultdict(float)

    denom = np.log(no_of_words_in_docs_with_label + (V * smoothing))
    for (word, _) in vocab:
        log_phi[(label, word)] = np.log(count_aggregator_for_label[word] + smoothing) - denom

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
    label_counter = Counter()
    label_counter.update(y)

    vocab_counter = Counter()

    for xi in x:
        vocab_counter.update(xi)

    vocab = set(vocab_counter.items())

    unique_labels = set(y)

    V = len(vocab)                           # vocab_size
    K = len(unique_labels)                   #

    # Need theta = [theta_1; theta_2; theta_3; ..., theta_i, ..., theta_K]
    # theta_i = [log_phi_i_1, log_phi_i_2, log_phi_i_3, ..., log_phi_i_i, ..., log_phi_i_V, 1].T

    theta = defaultdict(float)

    for label in unique_labels:
        theta.update(estimate_pxy_2(x, y, label, smoothing, vocab))
        mu = label_counter[label]/len(y)
        theta.update({(label, OFFSET): np.log(mu) })

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

    labels = list(set(y_tr))

    best_acc = 0
    best_smoother = None
    scores = {}

    for smoother in smoothers:
        theta_i = estimate_nb(x_tr, y_tr, smoother)
        y_hat = clf_base.predict_all(x_dv, theta_i, labels)
        acc = evaluation.acc(y_hat, y_dv)
        scores[smoother] = acc
        if acc > best_acc:
            best_acc = acc
            best_smoother = smoother

    return best_smoother, scores
