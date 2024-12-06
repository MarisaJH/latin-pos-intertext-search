# %%
# This notebook creates a word2vec model using the Bamman 2012 corpus lemmatized with the CLTK BackoffLatinLemmatizer

# %%
# Imports

import html
import string
import re
import os
import glob
import time
import multiprocessing

import collections

import gensim
from gensim.models import Word2Vec, FastText

#from cltk.stem.latin.j_v import JVReplacer
#from cltk.tokenize.sentence import TokenizeSentence
from cltk.sentence.lat import LatinPunktSentenceTokenizer
from nltk.tokenize import PunktSentenceTokenizer
#from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
from cltk.lemmatize.lat import LatinBackoffLemmatizer as BackoffLatinLemmatizer

#from matplotlib import pyplot

from pprint import pprint
import pickle

from nltk import word_tokenize

# %%
class JVReplacer:  # pylint: disable=too-few-public-methods
    """Replace J/V with I/U.
    Latin alphabet does not distinguish between J/j and I/i and V/v and U/u;
    Yet, many texts bear the influence of later editors and the predilections of other languages.

    In practical terms, the JV substitution is recommended on all Latin text preprocessing; it
    helps to collapse the search space.

    >>> replacer = JVReplacer()
    >>> replacer.replace("Julius Caesar")
    'Iulius Caesar'

    >>> replacer.replace("In vino veritas.")
    'In uino ueritas.'

    """

    def __init__(self):
        """Initialization for JVReplacer, reads replacement pattern tuple."""
        patterns = [(r"j", "i"), (r"v", "u"), (r"J", "I"), (r"V", "U")]
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        """Do j/v replacement"""
        for pattern, repl in self.patterns:
            text = re.subn(pattern, repl, text)[0]
        return text

# %%
# Set up NLP tools

replacer = JVReplacer()
tokenizer = LatinPunktSentenceTokenizer() #PunktSentenceTokenizer()
lemmatizer = BackoffLatinLemmatizer()

# %%
# Preprocess texts

def preprocess(text):
        
    text = text.lower()
    text = replacer.replace(text) #Normalize u/v & i/j
    
    punctuation ="\"#$%&\'()*+,-/:;<=>@[\]^_`{|}~.?!«»"
    translator = str.maketrans({key: " " for key in punctuation})
    text = text.translate(translator)

    translator = str.maketrans({key: " " for key in '0123456789'})
    text = text.translate(translator)
    return text

# %%
#helper iterator class to process raw text and to handle file by file. Avoids memory issues. 
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    
    def __iter__(self):
        #tokenizer = LatinPunktSentenceTokenizer() #TokenizeSentence('latin')
        for fname in glob.glob(self.dirname + '/*.txt'):
            with open(fname, encoding='utf-8') as file:
                #sents = tokenizer.tokenize_sentences(file.read().replace('\n', ''))
                sents = file.readlines()
                sents = [[token[1] for token in lemmatizer.lemmatize(preprocess(sent).split())] for sent in sents]
                for sent in sents:
                    yield sent



# Build Latin word2vec on Bamman data

cores = multiprocessing.cpu_count()

latin_w2v_model = Word2Vec(MySentences("../models/data/cc100-latin"), vector_size = 300, min_count=100, workers=cores-1, epochs=1)

# %%
latin_w2v_model.save("../models/latin_w2v_bamman_lemma300_100_1")


