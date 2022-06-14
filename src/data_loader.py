"""
load the json files and perform

Authors:
    * Megan Dare, 2022
    * Sangeet Sagar, 2022
"""

import sys,os
import json
import torch
import pdb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import Dataset
from transformers import BertTokenizer



class Phoenix2014T(Dataset):
    def __init__(self, data_path, tokenizer_model):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self._init_data(data_path)
        
        
    def __getitem__(self, index):
        example = self.data[index]
        gloss = example['gloss']
        text =  example['text']
        
        # Tokenize gloss
        encoded_gloss = self.tokenizer(gloss,
                                        padding = "max_length",
                                        return_attention_mask = True, 
                                        return_tensors = "pt")
        encoded_gloss_ids = encoded_gloss["input_ids"]
        encoded_gloss_attn_mask = encoded_gloss["attention_mask"]
        
        # Tokenize text
        encoded_text = self.tokenizer(text,
                                        padding = "max_length",
                                        return_attention_mask = True, 
                                        return_tensors = "pt")
        encoded_text_ids = encoded_text["input_ids"]
        encoded_text_attn_mask = encoded_text["attention_mask"]
        
        return encoded_gloss_ids. encoded_gloss_attn_mask, encoded_text_ids, encoded_text_attn_mask
    
    def __len__(self):
        return self.num_examples
    
    def _init_data(self, data_path):
        self.data = json.load(open(data_path))
        self.num_examples = len(self.data)
    

if __name__ == "__main__":
    
    hparams_file = sys.argv[1]
    # pdb.set_trace()
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    expt_dir = hparams["output_folder"]
    os.makedirs(expt_dir, exist_ok=True)
    test_set = Phoenix2014T(hparams['test'], hparams['tokenizer_model'])
    
    