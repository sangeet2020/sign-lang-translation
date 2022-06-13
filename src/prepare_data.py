from re import L
from transformers import AutoTokenizer
import torch


DATA_SRC_FOLDER = '../data/'
DEV_GLOSS = 'phoenix2014T.dev.gloss'
DEV_DE = 'phoenix2014T.dev.de'
TRAIN_GLOSS = 'phoenix2014T.train.gloss'
TRAIN_DE = 'phoenix2014T.train.de'
TEST_GLOSS = 'phoenix2014T.test.gloss'
TEST_DE = 'phoenix2014T.test.de'
VOCAB_GLOSS = 'phoenix2014T.vocab.gloss'
VOCAB_DE = 'phoenix2014T.vocab.de'

# load data from files
dev_gloss_lines = open(DATA_SRC_FOLDER + DEV_GLOSS, "r").read().split('\n')
dev_de_lines = open(DATA_SRC_FOLDER + DEV_DE, "r").read().split('\n')

'''
def make_pairs(source, target):
    pairs = []
    for i in range(0, len(source)):
        pairs.append((source[i], target[i]))
    return pairs

print(make_pairs(dev_gloss_lines, dev_de_lines))
'''

# tokenize using pretrained german bert tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

dev_gloss_encodings = tokenizer(dev_gloss_lines, truncation=True, padding=True, return_tensors='pt')
dev_de_encodings = tokenizer(dev_de_lines, truncation=True, padding=True, return_tensors='pt')


class Phoenix2014TDataset(torch.utils.data.Dataset):
    def __init__(self, gloss, text, tokenizer):
        assert len(gloss) == len(text), 'input and output sequences not of same length, this is unexpected'

        self.gloss = gloss
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._init_and_tokenize()

    def __getitem__(self, idx):
        # what should be returned by this function ?
        return self.data_gloss[idx]

    def __len__(self):
        return len(self.gloss)

    def _init_and_tokenize(self):
        self.data_gloss = self.tokenizer(self.gloss, 
                                         truncation=True, 
                                         padding=True, 
                                         return_tensors='pt',)
        self.data_de = self.tokenizer(self.text,
                                      truncation=True, 
                                      padding=True, 
                                      return_tensors='pt',)
