from gtnlplib.constants import OFFSET
import numpy as np
import torch
from collections import defaultdict


# deliverable 6.1
def get_top_features_for_label_numpy(weights, label, k=5):
    """
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in
    :param k: the number of top features to return. defaults to 5
    :returns: list of tuples of features and weights
    :rtype: list
    """

    weight_feature_list = list(weights)
    weight_key_list = [i for i in weight_feature_list if i[0] == label]
    weight_value_list = [weights[i] for i in weight_key_list]

    if len(weight_key_list) > k:
        indices = np.argpartition(weight_value_list, -k)[-k:]
        final_list = []
        for it in indices:
            final_list.append((weight_key_list[it], weights[weight_key_list[it]]))
    else:
        final_list = [(w, weights[w]) for w in weight_key_list ]
    final_list.sort(key=lambda x: x[1], reverse=True)
    return final_list


# deliverable 6.2
def get_top_features_for_label_torch(model, vocab, label_set, label, k=5):
    """
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    """

    vocab = sorted(vocab)

    weights = list(model.parameters())[0].data.numpy()

    indices = np.argpartition(weights[label_set.index(label), :], -k)[-k:]
    final_list = []
    for it in indices:
        final_list.append((vocab[it], weights[label_set.index(label), it]))

    final_list.sort(key=lambda x: x[1], reverse=True)

    return [i[0][0] for i in final_list]

    # raise NotImplementedError


# deliverable 7.1
def get_token_type_ratio(counts):
    """
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    """
    
    raise NotImplementedError


# deliverable 7.2
def concat_ttr_binned_features(data):
    """
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    """
    
    raise NotImplementedError
