import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag

import nose
import numpy as np

from imp import reload
import gtnlplib as gtnlplib

from nose.tools import with_setup, eq_, assert_almost_equals, ok_
from gtnlplib.parsing import ParserState, TransitionParser, DepGraphEdge, train
from gtnlplib.utils import DummyCombiner, DummyActionChooser, DummyWordEmbedding, DummyFeatureExtractor, \
    initialize_with_pretrained, build_suff_to_ix, initialize_with_pretrained
from gtnlplib.data_tools import Dataset
from gtnlplib.constants import *
from gtnlplib.evaluation import compute_metric, fscore, dependency_graph_from_oracle, output_preds
from gtnlplib.feat_extractors import SimpleFeatureExtractor
from gtnlplib.neural_net import FFActionChooser, FFCombiner, VanillaWordEmbedding, BiLSTMWordEmbedding, LSTMCombiner, \
    LSTMActionChooser, SuffixAndWordEmbedding
import gtnlplib.parsing as parsing
import gtnlplib.data_tools as data_tools
import gtnlplib.constants as consts
import gtnlplib.evaluation as evaluation
import gtnlplib.utils as utils
import gtnlplib.feat_extractors as feat_extractors
import gtnlplib.neural_net as neural_net
import pickle


def train_parser(parser, optimizer, dataset, n_epochs=1, n_train_insts=1000, name="eng"):
    for epoch in range(n_epochs):
        model_name = name + "-bakeoff-" + str(epoch + 1) + ".model"
        print("Epoch {}".format(epoch + 1), "Model:", model_name)

        parser.train()  # turn on dropout layers if they are there
        parsing.train(dataset.training_data[:n_train_insts], parser, optimizer, verbose=True)

        print("Dev Evaluation")
        parser.eval()  # turn them off for evaluation
        parsing.evaluate(dataset.dev_data, parser, verbose=True)
        print("F-Score: {}".format(evaluation.compute_metric(parser, dataset.dev_data, evaluation.fscore)))
        print("Attachment Score: {}".format(evaluation.compute_attachment(parser, dataset.dev_data)))
        print("\n")
        print("Saving Model:")
        torch.save(parser.state_dict(), model_name)

def build_parser(DROPOUT, LSTM_NUM_LAYERS, word_to_ix, pretrained_embeds):
    # Predef

    TEST_EMBEDDING_DIM = 4
    WORD_EMBEDDING_DIM = 64
    STACK_EMBEDDING_DIM = 100
    NUM_FEATURES = 3

    # Build Model
    feat_extractor = SimpleFeatureExtractor()
    # BiLSTM word embeddings will probably work best, but feel free to experiment with the others you developed
    word_embedding_lookup = BiLSTMWordEmbedding(word_to_ix, WORD_EMBEDDING_DIM, STACK_EMBEDDING_DIM,
                                                num_layers=LSTM_NUM_LAYERS, dropout=DROPOUT)
    initialize_with_pretrained(pretrained_embeds, word_embedding_lookup)
    action_chooser = LSTMActionChooser(STACK_EMBEDDING_DIM * NUM_FEATURES, LSTM_NUM_LAYERS, dropout=DROPOUT)
    combiner = LSTMCombiner(STACK_EMBEDDING_DIM, num_layers=LSTM_NUM_LAYERS, dropout=DROPOUT)
    parser = TransitionParser(feat_extractor, word_embedding_lookup, action_chooser, combiner)

    return parser


def train(ETA, DROPOUT, LSTM_NUM_LAYERS, n_epochs, dataset, word_to_ix, pretrained_embeds, output_preds_filename,
          dev_data, name, model_file_name = None):

    parser = build_parser(DROPOUT, LSTM_NUM_LAYERS, word_to_ix, pretrained_embeds)
    optimizer = optim.SGD(parser.parameters(), lr=ETA, momentum=0.5, dampening=0, nesterov=True)

    if model_file_name:
        # Load Model
        parser.load_state_dict(torch.load(model_file_name))
        parsing.evaluate(dataset.dev_data, parser, verbose=True)
        print("F-Score: {}".format(evaluation.compute_metric(parser, dataset.dev_data, evaluation.fscore)))
        print("Attachment Score: {}".format(evaluation.compute_attachment(parser, dataset.dev_data)))
        print("\n")
    else:
        # Train model
        train_parser(parser, optimizer, dataset, n_epochs=n_epochs, n_train_insts=10000, name=name)

    # Evaluate
    print("Creating file: ", output_preds_filename)
    output_preds(output_preds_filename, parser, dev_data)
    # print("LS: ")
    # !ls
    # print("Downloading...", output_preds_filename)
    # files.download(output_preds_filename)


def train_norwegian():
    # Params
    bakeoff_ETA_0_nr = 0.01
    bakeoff_DROPOUT_nr = 0.5
    bakeoff_LSTM_NUM_LAYERS_nr = 1

    pretrained_embeds = pickle.load(open(PRETRAINED_EMBEDS_FILE, 'rb'))  # NOT DOING ANYTHING FOR NORWEGIAN
    nr_dataset = Dataset(NR_TRAIN_FILE, NR_DEV_FILE, NR_TEST_FILE)
    word_to_ix_nr = {word: i for i, word in enumerate(nr_dataset.vocab)}
    nr_dev_data = [i.sentence for i in nr_dataset.dev_data]

    train(ETA=bakeoff_ETA_0_nr, DROPOUT=bakeoff_DROPOUT_nr, LSTM_NUM_LAYERS=bakeoff_LSTM_NUM_LAYERS_nr, n_epochs=5,
          dataset=nr_dataset, word_to_ix=word_to_ix_nr, pretrained_embeds=pretrained_embeds,
          output_preds_filename="bakeoff-dev-nr.preds", dev_data=nr_dev_data, name="norweg")


def train_english(model_file_name=None):
    # Params
    bakeoff_ETA_0_en = 0.01
    bakeoff_DROPOUT_en = 0.5
    bakeoff_LSTM_NUM_LAYERS_en = 1

    pretrained_embeds = pickle.load(open(PRETRAINED_EMBEDS_FILE, 'rb'))
    en_dataset = Dataset(EN_TRAIN_FILE, EN_DEV_FILE, EN_TEST_FILE)
    word_to_ix_en = {word: i for i, word in enumerate(en_dataset.vocab)}
    en_dev_data = [i.sentence for i in en_dataset.dev_data]

    train(ETA=bakeoff_ETA_0_en, DROPOUT=bakeoff_DROPOUT_en, LSTM_NUM_LAYERS=bakeoff_LSTM_NUM_LAYERS_en, n_epochs=5,
          dataset=en_dataset, word_to_ix=word_to_ix_en, pretrained_embeds=pretrained_embeds,
          output_preds_filename="bakeoff-dev-en.preds", dev_data=en_dev_data, name="eng", model_file_name=model_file_name)


if __name__ == '__main__':
    train_english()
