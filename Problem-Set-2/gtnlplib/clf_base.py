from gtnlplib.constants import OFFSET
from collections import defaultdict
import operator

# use this to find the highest-scoring label
argmax = lambda x: max(x.items(), key=operator.itemgetter(1))[0]


def make_feature_vector(base_features, label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    feature_vector = defaultdict(float)

    for el in base_features:
        feature_vector[(label, el)] = base_features[el]

    feature_vector[(label, OFFSET)] = 1
    return feature_vector
    

def predict(base_features, weights, labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = {}

    for label in labels:
        scores[label] = 0

    for label in labels:
        for feature in base_features:
            scores[label] += weights[(label, feature)] * base_features[feature]
        scores[label] += weights[(label, OFFSET)] * 1

    scores = dict(scores)
    return argmax(scores), scores
