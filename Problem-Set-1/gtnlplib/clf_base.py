from gtnlplib.constants import OFFSET
import numpy as np
from collections import defaultdict


# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]


# This will no longer work for our purposes since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

# deliverable 2.1
def make_feature_vector(base_features, label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict
    '''

    feature_vector = {}

    for el in base_features:
        feature_vector[(label, el)] = base_features[el]

    feature_vector[(label, OFFSET)] = 1
    return feature_vector


# deliverable 2.2
def predict(base_features, weights, labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict
    '''

    # print("TYPES:", base_features, "\nWeights:", weights, "\nLabels:", labels)

    scores = {}

    for label in labels:
        scores[label] = 0

    for label in labels:
        for feature in base_features:
            # print("Label: ", label, "Feature:", feature)
            scores[label] += weights[(label, feature)] * base_features[feature]
        scores[label] += weights[(label, OFFSET)] * 1

    scores = dict(scores)
    # print("\n\nscores: ", scores)
    return argmax(scores), scores


def predict_all(x, weights, labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = np.array([predict(x_i, weights, labels)[0] for x_i in x])
    return y_hat
