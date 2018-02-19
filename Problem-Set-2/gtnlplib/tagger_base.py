from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib import clf_base
from gtnlplib import bilstm
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE, UNK
import operator
from collections import Counter, defaultdict

argmax = lambda x: max(x.items(), key=lambda y: y[1])[0]


def make_classifier_tagger(weights):
    """

    :param weights: a defaultdict of classifier weights
    :returns: a function that takes a list of words, and a list of candidate tags, and returns tags for all words
    :rtype: function

    """

    # print("weights:", weights)
    # collect all tags and words
    tag_set = set()
    word_set = set()
    for (u_tag, u_word) in weights:
        tag_set.add(u_tag)
        word_set.add(u_word)

    unique_tags_list = list(tag_set)
    unique_words_list = list(word_set)

    most_f_tag = argmax({u_tag: weights[(u_tag, OFFSET)] for u_tag in unique_tags_list})

    best_tag = defaultdict(lambda: most_f_tag)
    for u_word in unique_words_list:
        some_var = {(tag, u_word): weights[(tag, u_word)] for tag in unique_tags_list}
        some_var.update({(tag, OFFSET): 1 for tag in unique_tags_list})
        best_tag[u_word] = argmax(some_var)[0]

    def classify(words, all_tags):
        """This nested function should return a list of tags, computed using a classifier with the weights passed as arguments to make_classifier_tagger and using basefeatures for each token (just the token and the offset)

        :param words: list of words
        :param all_tags: all possible tags
        :returns: list of tags
        :rtype: list

        """
        assigned_tags = []
        for word in words:
            assigned_tags.append(best_tag[word])
        return assigned_tags

    return classify


# compute tag with most unique word types: check if needed?
def most_unique_tag(weights, alltags):
    tag_uniq_counts = {tag: len([tup[0] for tup in weights.keys() if tup[0] == tag]) for tag in alltags}
    return argmax(tag_uniq_counts)


def apply_tagger(tagger, outfilename, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    if all_tags is None:
        all_tags = set()

        # this is slow
        for i, (words, tags) in enumerate(preproc.conll_seq_generator(trainfile)):
            for tag in tags:
                all_tags.add(tag)

    with open(outfilename, 'w') as outfile:
        for words, _ in preproc.conll_seq_generator(testfile):
            pred_tags = tagger(words, all_tags)
            for i, tag in enumerate(pred_tags):
                outfile.write(tag + '\n')
            outfile.write('\n')


def eval_tagger(tagger, outfilename, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger

    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurrences of (true_label, pred_label)
    """
    apply_tagger(tagger, outfilename, all_tags, trainfile, testfile)
    return scorer.get_confusion(testfile, outfilename)  # run the scorer on the prediction file


def apply_model(model, outfilename, word_to_ix, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    """
    applies the model on the data and writes the best sequence of tags to the outfile
    """
    if all_tags is None:
        all_tags = set()

        # this is slow
        for i, (words, tags) in enumerate(preproc.conll_seq_generator(trainfile)):
            for tag in tags:
                all_tags.add(tag)

    with open(outfilename, 'w') as outfile:
        for words, _ in preproc.conll_seq_generator(testfile):
            seq_words = bilstm.prepare_sequence(words, word_to_ix)
            pred_tags = model.predict(seq_words)
            for i, tag in enumerate(pred_tags):
                outfile.write(tag + '\n')
            outfile.write('\n')


def eval_model(model, outfilename, word_to_ix, all_tags=None, trainfile=TRAIN_FILE, testfile=DEV_FILE):
    """Calculate confusion_matrix for a given model

    Parameters:
    tagger -- Model mapping (words) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    apply_model(model, outfilename, word_to_ix, all_tags, trainfile, testfile)
    return scorer.get_confusion(testfile, outfilename)  # run the scorer on the prediction file
