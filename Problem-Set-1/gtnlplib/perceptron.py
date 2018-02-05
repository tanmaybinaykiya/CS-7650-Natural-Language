from collections import defaultdict
from gtnlplib.clf_base import predict, make_feature_vector


# deliverable 4.1
def perceptron_update(x, y, weights, labels):
    """
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features
    :param y: label, strings
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """

    # update = f(x, y) - f(x, y_real)
    y_predicted, y_score = predict(x, weights, labels)

    update = defaultdict(float)

    f_predicted = make_feature_vector(x, y_predicted)
    f_real = make_feature_vector(x, y)

    features = set(f_predicted.keys())
    features = features.union(f_real.keys())

    for features in list(features):
        value = f_real[features] - f_predicted[features]
        if value != 0:
            update[features] = value

    return update

# deliverable 4.2
def estimate_perceptron(x, y, N_its):
    """
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i, y_i in zip(x,y):
            dW = perceptron_update(x_i, y_i, weights, labels)
            for dw_i in dW.keys():
                weights[dw_i] += dW[dw_i]
        weight_history.append(weights.copy())
    return weights, weight_history

