# tokenizer for captions

import torch
import os
import json
import numpy as np
from collections import Counter
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt') # if not installed on your local, do this for tokenizer to work
# TODO: maybe use pickle to save the vocab to speed up the time



class Tokenizer():
    # Load datasets, also create vocab for tokenizer
    def __init__(self, caption_dir, MIN_FREQ=10, RM_TOP=5):
        with open(caption_dir) as json_file:
            caption_dict = json.load(json_file)

        text = ''.join(caption_dict.values())
        sentences = self.tokenize(text)
        vocab = [tokens for sent in sentences for tokens in sent]
        vocab_counts = Counter(vocab)
        stopwords = set([s[0] for s in vocab_counts.most_common(RM_TOP)])
        self.vocab = set([v for v in set(vocab) if vocab_counts[v] >= MIN_FREQ and v not in stopwords] + 
            ['<eos>'] + ['<pad>'] + ['<unk>'])
        
        self.vocab_size = len(self.vocab)
        self.w2i = {w: i for i, w in enumerate(sorted(self.vocab))}
        self.i2w = {i: w for i, w in enumerate(sorted(self.vocab))}
        self.eos = self.w2i['<eos>']
        self.pad = self.w2i['<pad>']
        self.unk = self.w2i['<unk>']
    
    def encode(self, input_text):
        """ encode the input using established vocab and dictionary """
        sentences = self.tokenize(input_text)
        output = []
        for sent in sentences:
            for tokens in sent:
                if tokens not in self.w2i:
                    output.append(self.unk)
                else:
                    output.append(self.w2i[tokens])
        return output
         
    def decode(self, input_token):
        return [self.i2w[token] for token in input_token]

    def tokenize(self, text):
        """
        Simple tokenizer (sentences and then words)
        """
        sentences = sent_tokenize(text)
        examples = []
        for sentence in sentences:
            sentence = "".join(char for char in sentence if char not in punctuation)
            sentence = "".join(char for char in sentence if not char.isdigit())
            sentence = sentence.lower()
            tokens = word_tokenize(sentence)
            examples.append(tokens)
        return examples
       
    

if __name__ == '__main__':  
    # test tokenizer
    caption_dir = '/mnt/d/Github/MICaptioning/iu_xray/iu_xray_captions.json'
    tokenizer = Tokenizer(caption_dir)
    text = """
    COPD and chronic opacities more pronounced in the lower lung XXXX. 
    There is persistent mild elevation right hemidiaphragm. 
    There is suggestion of subtle patchy opacities in lower lung XXXX bilaterally. 
    This is XXXX to be similar to XXXX scan. The heart is normal. 
    The aorta is calcified and tortuous. The skeletal structures show scoliosis and arthritic changes."""
    token = tokenizer.encode(text)
    decode_out = tokenizer.decode(token)
    print(token)
    print(decode_out)
    


