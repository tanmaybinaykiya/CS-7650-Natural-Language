# If you change the location of dataset, then
# Make the appropriate changes to the following constants.

prepend_path = ''
# /prepend_path = '../'

TRAIN_FILE = prepend_path + 'data/en-ud-train.conllu'
DEV_FILE = prepend_path + 'data/en-ud-dev.conllu'
TEST_FILE_UNLABELED = prepend_path + 'data/en-ud-test-hidden.conllu'
TEST_FILE = prepend_path + 'data/en-ud-test.conllu'

NR_TRAIN_FILE = prepend_path + 'data/no_bokmaal-ud-train.conllu'
NR_DEV_FILE = prepend_path + 'data/no_bokmaal-ud-dev.conllu'
NR_TEST_FILE_UNLABELED = prepend_path + 'data/no_bokmaal-ud-test-hidden.conllu'
NR_TEST_FILE = prepend_path + 'data/no_bokmaal-ud-test.conllu'

OFFSET = '**OFFSET**'

START_TAG = '--START--'
END_TAG = '--END--'

UNK = '<UNK>' 
