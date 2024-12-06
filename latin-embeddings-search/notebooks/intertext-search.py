# %%
# Imports

import os
import glob
import sys
import datetime

import re
from itertools import combinations, product
from functools import lru_cache
from statistics import mean

from natsort import natsorted

import pandas as pd
import numpy as np
import scipy.stats as ss
import Levenshtein

from gensim.models import Word2Vec

#from cltk.stem.latin.j_v import JVReplacer
#from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer
from cltk.lemmatize.lat import LatinBackoffLemmatizer as BackoffLatinLemmatizer

from pprint import pprint
import urllib
from tqdm import tqdm
import pickle

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
# Constants

TEXTS_FOLDER = "../data/texts/process"

files = natsorted(glob.glob(f'{TEXTS_FOLDER}/*.tess'))
MODEL = '../models/latin_w2v_cc100_lemma300_100_1'#'../models/latin_w2v_bamman_lemma300_100_1'
VECTORS = Word2Vec.load(MODEL).wv


lemmatize = True
lemmatize_bert_queries = True
lemmatizer = BackoffLatinLemmatizer()
replacer = JVReplacer()

# Create a new dictionary with replaced keys
new_key_to_index = {}
new_vectors = []
new_index_to_key = []

# Iterate through existing keys in order
for i, key in enumerate(VECTORS.index_to_key):
    new_key = replacer.replace(key)
    new_key_to_index[new_key] = i
    new_index_to_key.append(new_key)
    new_vectors.append(VECTORS.vectors[i])

# Update the model attributes
VECTORS.key_to_index = new_key_to_index
VECTORS.index_to_key = new_index_to_key
VECTORS.vectors = np.array(new_vectors)

# %%
# Get comparison dataset

#comps = pd.read_csv('../data/datasets/comps.csv', index_col=0)
if lemmatize_bert_queries:
    comps = pd.read_csv('../data/datasets/comps_bert_lemmas.csv', index_col=0)
    comps.dropna(subset=['bert_query', 'bert_result'], inplace=True)
    comps['bert_query'] = comps['bert_query'].apply(lambda x: tuple(x.split()))
    comps['bert_result'] = comps['bert_result'].apply(lambda x: tuple(x.split()))

    #print(comps.iloc[0])

    # lemmatize if not already lemmatized
    if not 'bert_query_lemmas' in comps.columns:
        comps['bert_query_lemmas'] = comps['bert_query'].apply(lambda x: [lemmatizer.lemmatize([word])[0][1] for word in x])
        comps['bert_target_lemmas'] = comps['bert_result'].apply(lambda x: [lemmatizer.lemmatize([word])[0][1] for word in x])

        print("after lemmatization")
        #print(comps.iloc[0])

        # join lemmas
        comps['bert_query_lemmas'] = comps['bert_query_lemmas'].apply(lambda x: f'{x[0]} {x[1]}')
        comps['bert_target_lemmas'] = comps['bert_target_lemmas'].apply(lambda x: f'{x[0]} {x[1]}')

        print("after joining lemmas")
        #print(comps.iloc[0])
        # save
        comps.to_csv('../data/datasets/comps_bert_lemmas.csv')
    
    comps['bert_query_lemmas'] = comps['bert_query_lemmas'].apply(lambda x: eval(x))
    comps['bert_target_lemmas'] = comps['bert_target_lemmas'].apply(lambda x: eval(x))

else:
    comps = pd.read_csv('../data/datasets/comps_bert_lemmas.csv', index_col=0)
    comps['query_lemma'] = comps['query_lemma'].apply(lambda x: eval(x))
    comps['target_lemma'] = comps['target_lemma'].apply(lambda x: eval(x))



# %%
# Text functions

def preprocess(text):
    replacer = JVReplacer()    
    text = text.lower()
    text = replacer.replace(text) #Normalize u/v & i/j
    
    text = text.replace('ego---sed', 'egosed') # handle a tesserae text issue
    
    punctuation ="\"#$%&\'()*+,-/:;<=>@[\]^_`{|}~.?!«»"
    translator = str.maketrans({key: " " for key in punctuation})
    text = text.translate(translator)

    translator = str.maketrans({key: " " for key in '0123456789'})
    text = text.translate(translator)
    
    text = text.replace('â\x80\x94', ' ')
    
    text = " ".join(text.split())
    
    return text

def index_tess(text):
    
    textlines = text.strip().split('\n')
    # https://stackoverflow.com/a/61436083
    def splitkeep(s, delimiter):
        split = s.split(delimiter)
        return [substr + delimiter for substr in split[:-1]] + [split[-1]]
    textlines = [splitkeep(line, '>') for line in textlines if line]
    return dict(textlines)

def pp_tess(tess_dict):
    return {k: preprocess(v) for k, v in tess_dict.items()}

def text_lemmatize(lemma_pairs):
    return " ".join([lemma for _, lemma in lemma_pairs])

def lem_tess(tess_dict):
    return {k: text_lemmatize(lemmatizer.lemmatize(v.split())) for k, v in tess_dict.items()}

def make_tess_file(str):
    str = str.lower()
    str = str.replace('lucan', 'lucan bellum_civile')
    str = str.split('.')[0]
    str = str.replace(' ', '.', 1)
    str = str.replace(' ', '.part.', 1)
    str += '.tess'
    return str

def make_tess_index(str):
    str = str.replace('Lucan', 'luc.').replace('Ovid', 'ov.').replace('Statius', 'stat.').replace('Vergil', 'verg.')
    str = str.replace('Metamorphoses', 'met.').replace('Thebaid', 'theb.').replace('Aeneid', 'aen.')
    str = str.split('-')[0]
    str = f'<{str}>'
    return str

def get_next_tess_index(index, n):
    index = index.replace('>','')
    index_base = index.split()[:-1]
    index_ref = index.split()[-1]
    index_ref_parts = index_ref.split('.')
    index_ref_next = int(index_ref_parts[1])+n
    next_index = f'{" ".join(index_base)} {index_ref_parts[0]}.{index_ref_next}>'
    
    exceptions = ['<luc. 1.419>', '<luc. 7.855>', '<luc. 7.856>', '<luc. 7.857>', '<luc. 7.858>', '<luc. 7.859>', '<luc. 7.860>', '<luc. 7.861>', '<luc. 7.862>', '<luc. 7.863>', '<luc. 7.864>', '<luc. 9.414>',
                  '<ov. met. 4.769>', 
                  '<stat. theb. 6.184>', '<stat. theb. 6.227>', '<stat. theb. 6.228>', '<stat. theb. 6.229>', '<stat. theb. 6.230>', '<stat. theb. 6.231>', '<stat. theb. 6.232>', '<stat. theb. 6.233>', '<stat. theb. 9.760>']
    if next_index in exceptions: # Handle missing data
        return None
    else:
        return next_index

# %%
# Get ngrams

def generate_ngrams(words_list, n):
    # Cf. https://www.techcoil.com/blog/how-to-generate-n-grams-in-python-without-using-any-external-libraries/
    ngrams_list = []

    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
    
    ngrams_list = [item.split() for item in ngrams_list if len(item.split()) == n]
    
    return ngrams_list

def generate_ngrams_interval(words_list, index, n, tess_dict, interval):
    
    limit = len(words_list) + interval + 1 # add one to avoid interval-based fencepost problem

    while len(words_list) < limit:
        words_extend = get_next_tess_index(index, 1)
        if words_extend:
            words_list += tess_dict[words_extend].split()
        else:
            break
    
    words_list = words_list[:limit]
    
    ngrams_list = []

    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
    
    ngrams_list = [item.split() for item in ngrams_list if len(item.split()) == n]
    
    return ngrams_list

def ngram_tess(tess_dict, n=2, interval=0):
    return {k: generate_ngrams_interval(v.split(), k, n, tess_dict, interval) for k, v in list(tess_dict.items())[:-1]} # Stop short of last item because of ngram lookahead

# %%
# Similarity functions

@lru_cache(maxsize=10000)
def get_similarities(terms, model):
    sims = []
    terms = list(terms)
    terms_ = set([y for x in terms for y in x])
    #oov = [term for term in terms_ if term not in model.vocab]
    oov = [term for term in terms_ if term not in model.key_to_index.keys()]

    for term in terms:
        if term[0] in oov or term[1] in oov:
            sim = -1
        else:
            sim = model.similarity(term[0], term[1])
        sims.append(sim)

    return sims

# pair-aware mean
def mean_similarities(terms, sims):
    max_sim_index = sims.index(max(sims))
    if max_sim_index == 0:
        pair_index = 3
    elif max_sim_index == 1:
        pair_index = 2
    elif max_sim_index == 2:
        pair_index = 1
    elif max_sim_index == 3:
        pair_index = 0
    return (sims[max_sim_index] + sims[pair_index])/2

sim_col = '_w2v'
if lemmatize_bert_queries:
    sim_col += '_bert-lemmas'
# calculate gold similarities
if ((lemmatize_bert_queries and not 'similarities_w2v_bert-lemmas' in comps.columns) or
    (not lemmatize_bert_queries and not 'similarities_w2v' in comps.columns)):
    print('Calculating gold similarities...')
    gold_similarities = []
    gold_similarities_sum = []

    for i, row in tqdm(comps.iterrows(), total=comps.shape[0]):  
        if lemmatize_bert_queries:
            query_pair = row['bert_query_lemmas']
            target_pair = row['bert_target_lemmas']
        else:
            query_pair = row['query_lemma']
            target_pair = row['target_lemma']
        
        all_pairs = tuple(product(query_pair, target_pair))
        similarities = get_similarities(all_pairs, VECTORS)
        gold_similarities.append(similarities)
        gold_similarities_sum.append(mean_similarities(all_pairs, similarities))

    # add to comps
    comps['similarities' + sim_col] = gold_similarities
    comps['similarity' + sim_col] = gold_similarities_sum

    # save comps
    comps.to_csv('../data/datasets/comps_bert_lemmas.csv')


# %%
# %%
# # Uncomment code to run full intertext search over texts

# # Get intertext search results
query_col = 'query_lemma'
if lemmatize_bert_queries:
    query_col = 'bert_query_lemmas'

results_ = []
start_idx, end_idx = 690, len(comps)
print(f"Processing rows {start_idx} to {end_idx}")
print('Using lemmatized bert queries' if lemmatize_bert_queries else 'Using original lemmas')
#for i, row in tqdm(comps.iterrows(), total=comps.shape[0]):    
for i, row in tqdm(comps.iloc[start_idx:end_idx].iterrows(), total=end_idx - start_idx):    
    #print(i, row['VF: Lemma'])
    #print('  gold sim:', row['similarity' + sim_col])
    search_files = natsorted([file for file in files if row['intertext_author'] in file])
    
    if np.isnan(row['interval']):
        interval = 0
    else:
        interval = int(row['interval'])
    
    n = row['query_length'] + interval
    
    results = []
    
    for file in search_files:
        with open(file, 'r') as f:
            contents = f.read()
            tess_dict = index_tess(contents)    
            tess_dict = pp_tess(tess_dict)
            if lemmatize:
                tess_dict = lem_tess(tess_dict)
        tess_dict = ngram_tess(tess_dict, n, interval)           

        for item in list(tess_dict.items()):            
            index = item[0]
            ngrams = item[1]
            for ngram in ngrams:                
                orderfree = row['orderfree']
                if orderfree:
                    combs = list(combinations(ngram, 2))
                else:
                    combs = [ngram]

                for comb in combs:
                    pairs = tuple(product(row[query_col], comb))
                    #print('  pairs:', pairs)
                    dists = get_similarities(pairs, VECTORS)
                    #print('  dists:', dists)
                    dists_sum = mean_similarities(pairs, dists)
                    
                    if dists_sum >= row["similarity" + sim_col]:
                        #print('dists_sum >= gold sim')
                        #print('  pairs:', pairs)
                        #print('  dists:', dists)
                        #print('  dists_sum:', dists_sum)
                        results.append((index, dists_sum, row[query_col], list(comb)))
    results_.append((row['index'], results))

# %%


# %%
# Create time-stamped file for results; cf. https://stackoverflow.com/a/14115286
output_path = f"{os.path.join('temp', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}-results-start-{start_idx}-end-{end_idx}-w2v.p"
if lemmatize_bert_queries:
    output_path = output_path.replace('.p', '_bert-lemmas.p')
pickle.dump(results_, open(output_path, 'wb'))

# For paper results, uncomment to download and run remaining cells
# url = 'https://utexas.box.com/shared/static/2m2e09dijfxgqg3nnxei2cuttc7qu7kd.p'
# urllib.request.urlretrieve (url, 'temp/results_naacl2021_search.p')

# output_path = 'temp/results_naacl2021_search.p'
# results_ = pickle.load(open(output_path, 'rb'))
print(f'Temp results saved at {output_path}')

# load results
'''
result_name = '../notebooks/temp/2024-11-24_03-32-36-results.p'
with open(result_name, 'rb') as f:
    results_ = pickle.load(f)
print('Results length:', len(results_))
'''
# %%
# Get ranks

ranks = [len(result[1]) for result in results_ if len(result[1]) != 0]

# %%
# Compute recall & precision at k; computer MRR

ks = [1, 3, 5, 10, 25, 50, 75, 100, 250]

def recall_at_k(ranks, k):
    n = len([rank for rank in ranks if rank <= k])
    d = len(ranks)
    recall = n/d
    return recall

def precision_at_k(ranks, k):
    n = len([rank for rank in ranks if rank <= k])
    d = sum([rank if rank<=k else k for rank in ranks])
    precision = n/d
    return precision

def mrr(ranks):
    return mean([1/item for item in ranks])

print(f'MRR: {mrr(ranks)}')
print()

print(f'Checking the following values for k {ks}\n')
for k in ks:
    print(f'\tRecall at k={k}: {recall_at_k(ranks, k)}')
    print(f'\tPrecision at k={k}: {precision_at_k(ranks, k)}')
    print()


