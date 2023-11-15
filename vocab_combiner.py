# coding=utf-8

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import os
import json

def combine_tokenizers(model1='bert-base-multilingual-cased', model2='bert-base-cased',save_path='models/tmp'):
    tok1 = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    vocab1 = tok1.vocab
    tok2 = AutoTokenizer.from_pretrained("bert-base-cased")
    vocab2 = tok2.vocab
    to_add = {}
    for key, value in vocab2.items():
        if key not in vocab1.keys():
            to_add[key] = value

    print("Found {} tokens to add".format(len(to_add)))

    mapping_dict_t1 = {}
    mapping_dict_t2 = {}

    tok1.add_tokens(list(to_add.keys()))    
    tok1.save_pretrained(save_path)
    for i in range(len(tok1.vocab)):
        mapping_dict_t1[i] = i
    for i in range(len(tok2.vocab)):
        mapping_dict_t2[i] = tok1.convert_tokens_to_ids((tok2.convert_ids_to_tokens(i)))

    with open(os.path.join(save_path,"mapping_t1.json"), "w") as outfile: 
        json.dump(mapping_dict_t1, outfile)
    with open(os.path.join(save_path,"mapping_t2.json"), "w") as outfile: 
        json.dump(mapping_dict_t2, outfile)

    print('Old vocab size:{}'.format(tok1.vocab_size))
    print('New vocab size: {}'.format(len(tok1.vocab)))

if __name__ == '__main__':
    combine_tokenizers()