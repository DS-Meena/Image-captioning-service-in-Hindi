"""
WSGI config for ICS_hindi project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

# ---------------------TEXT Vocab--------------------------------

import torchtext
from torchtext.data import get_tokenizer   # for tokenization
from collections import Counter     # for tokenizer

class textVocab:
    # method to initialize the variables
    def __init__(self):
        # MAIN DICT
        # create dict (int to word) (like token counter)
        self.itos = {0:"<PAD>", 1:"<start>", 2:"<end>", 3:"<UNK>"}
        
        # create dict (word to int) 
        self.stoi = {b:a for a, b in self.itos.items()}   
        
        # initialize word freq threshold
        self.min_freq = 1
        
        # MAIN COMPONENTS
        # tokenizer
        self.tokenizer = get_tokenizer("basic_english")   # works for HINDI also

        # token counter
        self.token_counter = Counter()
        
    # method to get size of vocabulary
    def __len__(self):
        return len(self.itos)
    
    # method to tokenize sentence
    def tokenize(self, text):
        return self.tokenizer(text)
    
    # method to numericalize sentence
    def numericalize(self, text):
        
        # tokenize the sentence
        tokens_list = self.tokenize(text)
        
        ans = []
        # convert words into ints (using stoi)
        for token in tokens_list:
            if token in self.stoi.keys():
                ans.append(self.stoi[token]) 
            else:
                ans.append(self.stoi["<UNK>"])
        return ans
#         return [ self.stoi[token] if token in self.stoi.keys() else self.stoi["<UNK>"] for token in tokens_list]
    
    # method to add new sentences to dict
    def build_vocab(self, sentence_list):
        word_count = 4
        
        # for each sentence
        for sentence in sentence_list:
            
            # tokenize
            tokens = self.tokenizer(sentence)
            
            # numericalize
            token_counter.update(tokens)
            
            # add words to vocab whose freq is >= min freq
            for token in tokens:
                if token_counter[token] >= self.min_freq and token not in self.stoi.keys():
                    self.stoi[token] = word_count
                    self.itos[word_count] = token
                    word_count += 1

# ---------------------------OLD Thing----------------------------------------

import os

from django.core.wsgi import get_wsgi_application

# for the text vocab error
# from static.text_pre_processing import textVocab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ICS_hindi.settings')

application = get_wsgi_application()
