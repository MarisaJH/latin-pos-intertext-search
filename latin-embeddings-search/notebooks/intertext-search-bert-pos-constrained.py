# %%
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import glob
from natsort import natsorted
from bert_embeddings import load_data
import re
import Levenshtein
import numpy as np
from itertools import product, combinations
import sys

# %%
DEBUG = True
TEXTS_FOLDER = "../data/texts/process"
EMBEDDINGS_FOLDER = "../data/bert_embeddings"

files = natsorted(glob.glob(f'{TEXTS_FOLDER}/*.tess'))
files_embeddings = natsorted(glob.glob(f'{EMBEDDINGS_FOLDER}/*.pos.pkl'))
MODEL = 'bowphs/LaBerta'
lemmatize = False

# %%
comps = pd.read_csv('../data/datasets/comps_paper.csv', index_col=0)

# %%
vf_file = EMBEDDINGS_FOLDER + "/valerius_flaccus.argonautica.part.1.tess.pos.pkl"
vf_data = load_data(vf_file)

# ------------------------------------------------------------
# CONSTRAINTS
# ------------------------------------------------------------
# Match both words in the bigram to these constraints
MATCH_TARGET_POS = False # filter out results that don't match the target POS
MATCH_TARGET_WHOLE_MORPH = False # filter out results that don't match the target whole morph
MATCH_QUERY_POS = False # filter out results that don't match the query POS
MATCH_QUERY_MORPH = False # filter out results that don't match the query morph

# Real world settings
MATCH_QUERY_FEAT = None # 'Voice'
NOT_MATCH_QUERY_FEAT = None # 'Case'

# which position in the bigram to apply the constraint to;
# (1,0) -> apply to first word in bigram
# (0,1) -> apply to second word in bigram
# (1,1) -> apply to both words in bigram
APPLY_TO_POS = (0,1)

FINEGRAINED = (
    'pos=PROPN|Gender=Masc',
    ''
)

# set to True if running on small set; for examples
print_results = True

# if None, use threshold of gold similarity (for measuring precision/recall/mrr);
# if set, use for printing the retrieved results
threshold = 0.3

# %%
def sort_data(data: dict):
    # reorder vf_data so that the lines are in order
    # sort by 'references' key
    new_references = sorted(data['references'], key=lambda x: int(x.split('.')[-1]))

    # reorder vf_data
    old_ref_to_idx = {ref: i for i, ref in enumerate(data['references'])}
    #new_data = deepcopy(data)
    #for i, ref in enumerate(new_references):
    #    new_data['references'][i] = ref
    #    new_data['embeddings'][i] = data['embeddings'][old_ref_to_idx[ref]]
    #    new_data['sentences'][i] = data['sentences'][old_ref_to_idx[ref]]
    #    new_data['subword_tokens'][i] = data['subword_tokens'][old_ref_to_idx[ref]]
    #    new_data['word_tokens'][i] = data['word_tokens'][old_ref_to_idx[ref]]
    #    new_data['word_to_tokens_maps'][i] = data['word_to_tokens_maps'][old_ref_to_idx[ref]]

    # ref -> {key: value}
    new_data = {}
    for i, ref in enumerate(new_references):
        new_data[ref] = {key: data[key][old_ref_to_idx[ref]] for key in data.keys()}
    return new_data

# %%
new_vf_data = sort_data(vf_data)

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
REPLACER = JVReplacer()

# %%
def clean_word(word: str):
    word = word.lower()
    word = REPLACER.replace(word)

    # special cases
    #if word in ['fatidicamque', 'mediosque']:
    #    word = word[:-3]
    return word


# %%
def search_for_word(word: str, word_tokens: list[list[str]]):
    strip_que = not(word.endswith('que') and len(word) > 3)
    strip_ne = not(word.endswith('ne') and len(word) > 2)
    
    for i, line in enumerate(word_tokens):
        for j, w in enumerate(line):
            if strip_que and w.endswith('que') and len(w) > 3:
                stripped_word = w[:-3]
                if stripped_word == word:
                    return i, j
            elif strip_ne and w.endswith('ne') and len(w) > 2:
                stripped_word = w[:-2]
                if stripped_word == word:
                    return i, j
            else:
                if w == word:
                    return i, j
    return -1, -1



# %%
def find_closest_word(word: str, alt_phrase: list[str]):
    # find the word in alt_phrase that has the smallest edit distance to word
    min_dist = float('inf')
    closest_word = None
    for alt_word in alt_phrase:
        dist = Levenshtein.distance(word, alt_word)
        if dist < min_dist:
            min_dist = dist
            closest_word = alt_word
    return closest_word, min_dist

# %%
def get_line_keys(data: dict, line_pos: tuple[int, int]):
    first_key = ''
    for key in data:
        first_key = key
        break

    *key_prefix, _ = first_key.split('.')
    key_prefix = '.'.join(key_prefix)

    start, end = line_pos
    keys = [f'{key_prefix}.{i}' for i in range(start, end+1)]
    # remove any keys that don't exist
    keys = [key for key in keys if key in data]
    return keys


# %%
def get_bigram_embeddings(bigram: tuple[str, str], alt_phrase: list[str], data: dict, line_pos: tuple[int, int]):
    ''' 
    bigram: tuple of two strings, the bigram to find in the text
    data: dictionary of embeddings for the Tesseract file
    line_pos: tuple of two ints, the start and end indices of the lines to search
    interval: int, the interval between the the words in the bigram
    '''
    start, end = line_pos
    #lines = data['references'][start:end]
    #print('refs:', lines)
    #embeddings = data['embeddings'][start:end]
    #word_tokens = [[clean_word(w) for w in words] for words in data['word_tokens'][start:end]]
    #subword_tokens = data['subword_tokens'][start:end]
    #word_to_tokens_maps = data['word_to_tokens_maps'][start:end]

    keys = get_line_keys(data, line_pos)
    embeddings = [data[key]['embeddings'] for key in keys]
    word_tokens = [data[key]['word_tokens'] for key in keys]
    word_tokens = [[clean_word(w) for w in words] for words in word_tokens]
    subword_tokens = [data[key]['subword_tokens'] for key in keys]
    word_to_tokens_maps = [data[key]['word_to_tokens_maps'] for key in keys]
    print('refs:', keys)
    word_idx_to_ref = {i: key for i, key in enumerate(keys)}

    print('word tokens')
    for i, line in enumerate(word_tokens):
        print(i, line)
    
    # find words in bigram
    w1_line_idx, w1_idx = search_for_word(bigram[0], word_tokens)
    w2_line_idx, w2_idx = search_for_word(bigram[1], word_tokens)

    if w1_line_idx == -1 and w2_line_idx != -1:
        if bigram[1] in alt_phrase: alt_phrase.remove(bigram[1])
        alt_w1, alt_w1_score = find_closest_word(bigram[0], alt_phrase)
        w1_line_idx, w1_idx = search_for_word(alt_w1, word_tokens)
        print(f'  set alt w1 from {bigram[0]} to', alt_w1)
    elif w2_line_idx == -1 and w1_line_idx != -1:
        if bigram[0] in alt_phrase: alt_phrase.remove(bigram[0])
        alt_w2, alt_w2_score = find_closest_word(bigram[1], alt_phrase)
        w2_line_idx, w2_idx = search_for_word(alt_w2, word_tokens)
        print(f'  set alt w2 from {bigram[1]} to', alt_w2)
    elif w2_line_idx == -1 and w1_line_idx == -1:
        alt_w1, alt_w1_score = find_closest_word(bigram[0], alt_phrase)
        alt_w2, alt_w2_score = find_closest_word(bigram[1], alt_phrase)

        # want to use the word with the smallest edit distance first,
        # then remove that word from alt_phrase and find the second word
        if alt_w1_score < alt_w2_score:
            w1_line_idx, w1_idx = search_for_word(alt_w1, word_tokens)
            alt_phrase.remove(alt_w1)
            alt_w2, alt_w2_score = find_closest_word(bigram[1], alt_phrase)
            w2_line_idx, w2_idx = search_for_word(alt_w2, word_tokens)
        else:
            w2_line_idx, w2_idx = search_for_word(alt_w2, word_tokens)
            alt_phrase.remove(alt_w2)
            alt_w1, alt_w1_score = find_closest_word(bigram[0], alt_phrase)
            w1_line_idx, w1_idx = search_for_word(alt_w1, word_tokens)

        print(f'  set alt w1 from {bigram[0]} to', alt_w1)
        print(f'  set alt w2 from {bigram[1]} to', alt_w2)


    if w1_line_idx == -1 or w2_line_idx == -1:
        print(f'WARNING: bigram {bigram} not found in lines {start} to {end}')
        return None
    print('w1_line_idx, w1_idx:', w1_line_idx, w1_idx)
    print('word tokens:', word_tokens[w1_line_idx])
    print('subword tokens:', subword_tokens[w1_line_idx])
    print('w2_line_idx, w2_idx:', w2_line_idx, w2_idx)
    print('word tokens:', word_tokens[w2_line_idx])
    print('subword tokens:', subword_tokens[w2_line_idx])
    
    # get embeddings
    w1_subword_idx = word_to_tokens_maps[w1_line_idx][w1_idx]
    w2_subword_idx = word_to_tokens_maps[w2_line_idx][w2_idx]
    print('w1_subword_idx:', w1_subword_idx)
    print('w2_subword_idx:', w2_subword_idx)

    w1_embedding = embeddings[w1_line_idx][w1_subword_idx]
    w2_embedding = embeddings[w2_line_idx][w2_subword_idx]

    # average subword embeddings
    w1_embedding = np.mean(w1_embedding, axis=0)
    w2_embedding = np.mean(w2_embedding, axis=0)
    
    if DEBUG: print(f'Given bigram: {bigram}, found words: {word_tokens[w1_line_idx][w1_idx]} and {word_tokens[w2_line_idx][w2_idx]}')

    w1 = word_tokens[w1_line_idx][w1_idx]
    w2 = word_tokens[w2_line_idx][w2_idx]

    # also return indices with line offset
    w1_global_line = int(word_idx_to_ref[w1_line_idx].split('.')[-1])
    w2_global_line = int(word_idx_to_ref[w2_line_idx].split('.')[-1])

    w1_final_idx = (w1_idx, w1_global_line)
    w2_final_idx = (w2_idx, w2_global_line)

    return (w1_embedding, w2_embedding), (w1, w2), (w1_final_idx, w2_final_idx)


# %%
def get_gold_bigram_phrase(row: pd.Series):
    vf_bigram_phrase = row['Query'] # not actually lemmas
    vf_bigram_phrase = REPLACER.replace(vf_bigram_phrase)
    vf_bigram_phrase = vf_bigram_phrase.lower().split()

    alt_phrase = row['VF: Lemma']
    alt_phrase = REPLACER.replace(alt_phrase)
    alt_phrase = alt_phrase.lower().split()
    while '...' in alt_phrase:
        alt_phrase.remove('...')

    return vf_bigram_phrase, alt_phrase


# %%
def get_intertext_phrase(row: pd.Series):
    intertext_phrase = row['Result']
    intertext_phrase = REPLACER.replace(intertext_phrase)
    intertext_phrase = intertext_phrase.lower().split()
    
    alt_phrase = row['Intertext: Phrase']
    alt_phrase = REPLACER.replace(alt_phrase)
    # strip out punctuation
    alt_phrase = ''.join(char for char in alt_phrase if char.isalpha())
    alt_phrase = alt_phrase.lower().split()
    return intertext_phrase, alt_phrase


# %%
def compute_bigram_similarity(similarities: np.ndarray):
    best_idx = np.argmax(similarities)
    sim_1 = similarities[best_idx]
    # get index of position that doesn't contain the same word
    if best_idx == 0:
        pair_index = 3
    elif best_idx == 1:
        pair_index = 2
    elif best_idx == 2:
        pair_index = 1
    elif best_idx == 3:
        pair_index = 0

    sim_2 = similarities[pair_index]
    final_sim = (sim_1 + sim_2) / 2
    return final_sim

# %% [markdown]
# calculate bigram similarities and save to file
'''
# %%
# cache of data, since we'll be accessing out of order
intertext_file_to_data = {}

# %%
# get embeddings
skip_idx = [67, 68, 525, 746, 839]
comps = pd.read_csv('../data/datasets/comps_paper.csv', index_col=0)

bert_query = []
bert_query_indices = []
bert_result = []
bert_result_indices = []
bert_pairs = []
bert_similarities = [] # 4 pair similarities per row
bert_similarity = [] # final similarities

for idx, row in comps.iterrows():
    if idx in skip_idx:
        bert_pairs.append(None)
        bert_similarities.append(None)
        bert_similarity.append(None)
        bert_query.append(None)
        bert_result.append(None)
        bert_query_indices.append(None)
        bert_result_indices.append(None)
        continue

    # get vf embeddings
    vf_bigram_phrase, alt_phrase = get_gold_bigram_phrase(row)
    
    start_line = row['VF: Line Start'] - 2
    if start_line < 1:
        start_line = 1
    end_line = start_line + 5

    if idx == 395:
        end_line += 3
    #line_pos = (start_idx, end_idx)
    line_pos = (start_line, end_line)
    #interval = row['Interval']
    if idx == 635:
        vf_bigram_phrase[0] = 'umme' # tokenization issue with roberta
    elif 870 <= idx <= 871:
        vf_bigram_phrase[0] = 'itte'
    
    vf_embeddings, vf_bigram, vf_bigram_indices = get_bigram_embeddings(vf_bigram_phrase, alt_phrase, new_vf_data, line_pos)

    # get intertext embeddings
    intertext_phrase, alt_phrase = get_intertext_phrase(row)
    intertext_file = row['file']
    intertext_data_file = EMBEDDINGS_FOLDER + f'/{intertext_file}.pos.pkl'

    if intertext_file not in intertext_file_to_data:
        intertext_data = load_data(intertext_data_file)

        # sort data by line number
        intertext_file_to_data[intertext_file] = sort_data(intertext_data)
    intertext_data = intertext_file_to_data[intertext_file]

    # get embeddings
    #start_idx = row['VF: Line Start'] - 1 # line start, not word start
    start_line = row['Intertext: Line Start'] - 2
    if start_line < 1:
        start_line = 1
    end_line = start_line + 5

    line_pos = (start_line, end_line)
    
    intertext_embeddings, intertext_bigram, intertext_bigram_indices = get_bigram_embeddings(intertext_phrase, alt_phrase, intertext_data, line_pos)

    # generate pairs
    pairs = list(product(vf_bigram, intertext_bigram))
    similarities = np.zeros((4,))
    indices = ((0, 0), (0, 1), (1, 0), (1, 1))
    for i, (vf_i, intertext_i) in enumerate(indices):
        vf_w, intertext_w = pairs[i]
        vf_emb = vf_embeddings[vf_i]
        intertext_emb = intertext_embeddings[intertext_i]
        # cosine similarity
        similarities[i] = np.dot(vf_emb, intertext_emb) / (np.linalg.norm(vf_emb) * np.linalg.norm(intertext_emb))

    # find highest similarity index
    final_sim = compute_bigram_similarity(similarities)

    bert_pairs.append(pairs)
    bert_similarities.append(similarities)
    bert_similarity.append(final_sim)
    bert_query.append(' '.join(vf_bigram))
    bert_result.append(' '.join(intertext_bigram))
    bert_query_indices.append(vf_bigram_indices)
    bert_result_indices.append(intertext_bigram_indices)


# %%
len(bert_pairs)

# %%
row = comps.iloc[0]
row

# %%
bert_query[0], bert_query_indices[0]

# %%
# add to dataframe
comps['bert_query'] = bert_query
comps['bert_result'] = bert_result
comps['bert_query_indices'] = bert_query_indices
comps['bert_result_indices'] = bert_result_indices
comps['bert_pairs'] = bert_pairs
comps['bert_similarities'] = bert_similarities
comps['bert_similarity'] = bert_similarity

comps.to_csv('../data/datasets/comps_bert.csv')
'''
# %% [markdown]
# intertext search over all possible bigrams

# %%
from tqdm import tqdm
from cltk.lemmatize.lat import LatinBackoffLemmatizer as BackoffLatinLemmatizer
lemmatizer = BackoffLatinLemmatizer()

# %%
lemmatize = False

# %%
comps = pd.read_csv('../data/datasets/comps_bert_pos.csv', index_col=0)
comps['bert_query'] = comps['bert_query'].apply(lambda x: x.split() if not pd.isna(x) else [])
comps['bert_result'] = comps['bert_result'].apply(lambda x: x.split() if not pd.isna(x) else [])


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
 
def get_next_tess_index(index, n, data):
    *key_prefix, i = index.split('.')
    key_prefix = '.'.join(key_prefix)
    i = int(i)
    i += n
    
    new_index = f'{key_prefix}.{i}'
    if new_index in data:
        return new_index
    else:
        return None



# %%
# Get ngrams
remove_tokens = ['<s>', '</s>', '<unk>', '<pad>']
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def generate_ngrams(words_list, n):
    # Cf. https://www.techcoil.com/blog/how-to-generate-n-grams-in-python-without-using-any-external-libraries/
    ngrams_list = []

    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
    
    ngrams_list = [item.split() for item in ngrams_list if len(item.split()) == n]
    
    return ngrams_list

def generate_ngrams_interval(words_list, index, n, tess_dict, interval):
    valid_indices = []
    for i, token in enumerate(words_list):
        if token not in remove_tokens and token not in punctuation:
            valid_indices.append((i, 0)) # (index, offset from current reference/index)
    
    limit = len(valid_indices) + interval + 1 # add one to avoid interval-based fencepost problem
    
    current_ref = index
    ref_offset = 0
    while len(valid_indices) < limit:
        next_ref = get_next_tess_index(current_ref, 1, tess_dict)
        if next_ref:
            ref_offset += 1
            start_idx = len(words_list)
            next_words = tess_dict[next_ref]['word_tokens']
            words_list.extend(next_words)
            
            # Add new valid indices
            for i, token in enumerate(next_words):
                if token not in remove_tokens and token not in punctuation:
                    valid_indices.append((i, ref_offset))
            current_ref = next_ref
        else:
            break
    valid_indices = valid_indices[:limit]
    
    ngrams_list = []
    for num in range(len(valid_indices) - n + 1):
        ngram_indices = valid_indices[num:num + n]
        if len(ngram_indices) == n:
            ngrams_list.append(ngram_indices)
    
    return ngrams_list

def ngram_tess(tess_dict, n=2, interval=0):
    '''Return ngram indices into the word_tokens list, skipping punctuation'''
    #return {k: generate_ngrams_interval(v.split(), k, n, tess_dict, interval) for k, v in list(tess_dict.items())[:-1]} # Stop short of last item because of ngram lookahead
    ngrams_dict = {}
    for ref, data in tess_dict.items():
        word_tokens = data['word_tokens']
        ngram_indices = generate_ngrams_interval(word_tokens, ref, n, tess_dict, interval)
        
        tokens = []
        for ngram in ngram_indices:
            ngram_tokens = []
            for idx, offset in ngram:
                if offset == 0:
                    token = word_tokens[idx]
                else:
                    next_ref = get_next_tess_index(ref, offset, tess_dict)
                    token = tess_dict[next_ref]['word_tokens'][idx]
                ngram_tokens.append(token)
            tokens.append(ngram_tokens)

        ngrams_dict[ref] = {
            'indices': ngram_indices,
            'tokens': tokens
        }

    return ngrams_dict



# %%
def get_similarities(pairs):
    return [np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)) for p1, p2 in pairs]


# %%
def get_bert_embeddings(indices: tuple[int, int], data: dict, line_is_global: bool = True, ref: str = None):
    '''If line_is_global, then indices are global line numbers, not offsets;
    otherwise, indices are offsets from ref
    
    indices: tuple[int, int], the word and line indices for a single word
    ''' 

    word_idx, line_idx = indices
    if line_is_global:
        first_key = ''
        for key in data:
            first_key = key
            break

        *key_prefix, _ = first_key.split('.')
        key_prefix = '.'.join(key_prefix)
        new_ref = f'{key_prefix}.{line_idx}'
    else:
        offset = line_idx
        *key_prefix, ref_i = ref.split('.')
        key_prefix = '.'.join(key_prefix)
        ref_i = int(ref_i)
        new_ref = f'{key_prefix}.{ref_i + offset}'

    # subword embeddings
    
    word_to_tokens_map = data[new_ref]['word_to_tokens_maps'][word_idx]
    #subword_tokens = [data[new_ref]['subword_tokens'][i] for i in word_to_tokens_map]
    #print('tokens:', subword_tokens)
    embeddings = data[new_ref]['embeddings'][word_to_tokens_map]
    #print('num embeddings:', len(embeddings))
    # average to get word embedding
    word_embedding = np.mean(embeddings, axis=0)
    
    return word_embedding

def get_pos_morph_tags(indices: tuple[int, int], data: dict, line_is_global: bool = True, ref: str = None):
    '''If line_is_global, then indices are global line numbers, not offsets;
    otherwise, indices are offsets from ref
    
    indices: tuple[int, int], the word and line indices for a single word
    ''' 
    word_idx, line_idx = indices
    if line_is_global:
        first_key = ''
        for key in data:
            first_key = key
            break

        *key_prefix, _ = first_key.split('.')
        key_prefix = '.'.join(key_prefix)
        new_ref = f'{key_prefix}.{line_idx}'
    else:
        offset = line_idx
        *key_prefix, ref_i = ref.split('.')
        key_prefix = '.'.join(key_prefix)
        ref_i = int(ref_i)
        new_ref = f'{key_prefix}.{ref_i + offset}'

    pos_tags = data[new_ref]['pos_tags'][word_idx]
    morph_tags = data[new_ref]['morph_tags'][word_idx]
    
    return pos_tags, morph_tags
# %%
# # Uncomment code to run full intertext search over texts

# # Get intertext search results
def matches_finegrained(constraint_str, w_pos: str, w_morph: dict):
    constraints = constraint_str.split('|')
    for constraint in constraints:
        feat, value = constraint.split('=')
        if feat == 'pos':
            if w_pos != value:
                #print(f'  {w_pos} != {value}')
                return False
        else:
            if w_morph.get(feat, '') != value:
                #print(f'  {w_morph.get(feat, "")} != {value}')
                return False
    #print('  MATCH')
    return True

def matches_constraints(comb_pos, comb_morph, query_pos, query_morph, result_pos, result_morph, orderfree):
    # order doesn't matter, so convert to sets
    if orderfree:
        comb_pos_set = set(comb_pos)
        result_pos_set = set(result_pos)
        query_pos_set = set(query_pos)
    
    if MATCH_TARGET_POS:
        if comb_pos_set != result_pos_set:
            return False
    if MATCH_TARGET_WHOLE_MORPH:
        if orderfree:
            matches = (comb_morph == result_morph) or (comb_morph[0], comb_morph[1]) == (result_morph[1], result_morph[0])
        else:
            matches = comb_morph == result_morph
        #print(f"  Checking target morph: {comb_morph} == {result_morph} -> {matches}")
        if not matches:
            return False
    if MATCH_QUERY_POS:
        if comb_pos_set != query_pos_set:
            return False
    if MATCH_QUERY_MORPH:
        if orderfree:
            matches = (comb_morph == query_morph) or (comb_morph[0], comb_morph[1]) == (query_morph[1], query_morph[0])
        else:
            matches = comb_morph == query_morph
        if not matches:
            return False
        
    finegrained_w1, finegrained_w2 = FINEGRAINED
    #print('comb_pos:', comb_pos)
    #print('comb_morph:', comb_morph)
    if finegrained_w1 and orderfree:
        if not (matches_finegrained(finegrained_w1, comb_pos[0], comb_morph[0]) or 
                matches_finegrained(finegrained_w1, comb_pos[1], comb_morph[1])):
            return False
    elif finegrained_w1 and not orderfree:
        if not matches_finegrained(finegrained_w1, comb_pos[0], comb_morph[0]):
            return False
    if finegrained_w2 and orderfree:
        if not (matches_finegrained(finegrained_w2, comb_pos[0], comb_morph[0]) or 
                matches_finegrained(finegrained_w2, comb_pos[1], comb_morph[1])):
            return False
    elif finegrained_w2 and not orderfree:
        if not matches_finegrained(finegrained_w2, comb_pos[1], comb_morph[1]):
            return False

    '''
    if NOT_MATCH_QUERY_FEAT and MATCH_QUERY_POS:
        check_w1, check_w2 = APPLY_TO_POS 
        if orderfree:
            # find comb with same POS as query word
            if check_w1:
                w1_i = comb_pos.index(query_pos[0])
            if check_w2:
                w2_i = comb_pos.index(query_pos[1])

            # now check morph
            if check_w1: 
                if comb_morph[0].get(NOT_MATCH_QUERY_FEAT, '') != query_morph[w1_i].get(NOT_MATCH_QUERY_FEAT, ''):
                    return False
            if check_w2:
                if comb_morph[1].get(NOT_MATCH_QUERY_FEAT, '') != query_morph[w2_i].get(NOT_MATCH_QUERY_FEAT, ''):
                    return False
        else:
            if check_w1:
                if comb_morph[0].get(NOT_MATCH_QUERY_FEAT, '') != query_morph[0].get(NOT_MATCH_QUERY_FEAT, ''):
                    return False
            if check_w2:
                if comb_morph[1].get(NOT_MATCH_QUERY_FEAT, '') != query_morph[1].get(NOT_MATCH_QUERY_FEAT, ''):
                    return False
    '''
    #print('RETURNING TRUE')
    return True

results_ = []

start_idx, end_idx = 255, 256
print(f'Processing rows {start_idx} to {end_idx}')

#for i, row in tqdm(comps.iterrows(), total=comps.shape[0]):    
for i, row in tqdm(comps.iloc[start_idx:end_idx].iterrows(), total=end_idx - start_idx):    
    search_files = natsorted([file for file in files_embeddings if row['intertext_author'] in file])
    
    if np.isnan(row['interval']):
        interval = 0
    else:
        interval = int(row['interval'])
    
    n = row['query_length'] + interval
    
    results = []
    if not row['bert_query']:
        continue

    # query bert embeddings 
    query_indices = eval(row['bert_query_indices'])
    vf_w1_embedding = get_bert_embeddings(query_indices[0], new_vf_data, line_is_global=True)
    vf_w2_embedding = get_bert_embeddings(query_indices[1], new_vf_data, line_is_global=True)
    query_embeddings = (vf_w1_embedding, vf_w2_embedding)

    # pos and morph tags for query
    query_pos = eval(row['query_pos'])
    query_morph = eval(row['query_morph'])

    # pos and morph tags for gold target
    result_pos = eval(row['result_pos'])
    result_morph = eval(row['result_morph'])

    target_toks = row['bert_result']
    
    for file in search_files:
        #with open(file, 'r') as f:
        #    contents = f.read()
        #    tess_dict = index_tess(contents)    
        #    tess_dict = pp_tess(tess_dict)
        #    if lemmatize:
        #        tess_dict = lem_tess(tess_dict)
        #tess_dict = ngram_tess(tess_dict, n, interval) 
        
        # open data file
        tess_data = load_data(file)
        tess_data = sort_data(tess_data)

        # generate ngrams
        tess_dict = ngram_tess(tess_data, n, interval)      

        for ref, ngram_data in tess_dict.items():           
            
            for j, ngram in enumerate(ngram_data['indices']): 
                tokens = ngram_data['tokens'][j]            
                orderfree = row['orderfree']
                if orderfree:
                    combs = list(combinations(ngram, 2))
                    combs_tokens = list(combinations(tokens, 2))
                else:
                    combs = [ngram]
                    combs_tokens = [tokens]

                for comb, comb_tokens in zip(combs, combs_tokens):
                    comb1_pos, comb1_morph = get_pos_morph_tags(comb[0], tess_data, line_is_global=False, ref=ref)
                    comb2_pos, comb2_morph = get_pos_morph_tags(comb[1], tess_data, line_is_global=False, ref=ref)
                    comb_pos = (comb1_pos, comb2_pos)
                    comb_morph = (comb1_morph, comb2_morph)

                    #if tuple(comb_tokens) == target_toks or (orderfree and (comb_tokens[0], comb_tokens[1]) == (target_toks[1], target_toks[0])):
                        #print('MATCHING TOKENS')
                        #print('  comb:', comb_tokens)
                        #print('  target:', target_toks)
                        #print('  comb_pos:', comb_pos)
                        #print('  target_pos:', result_pos)
                        #print('  comb_morph:', comb_morph)
                        #print('  target_morph:', result_morph)
                        #print('  matches_constraints:', matches_constraints(comb_pos, comb_morph, query_pos, query_morph, result_pos, result_morph))

                    # can skip similarity computation if constraints not met
                    if not matches_constraints(comb_pos, comb_morph, query_pos, query_morph, result_pos, result_morph, orderfree):
                        #print('NOT MATCHING CONSTRAINTS')
                        #print('  comb:', comb_pos, comb_morph)
                        #print('  query:', query_pos, query_morph)
                        #print('  result:', result_pos, result_morph)
                        #print('  comb tokens:', comb_tokens)
                        #print('  target tokens:', target_toks)
                        continue
                    #else:
                        #print('MATCHING CONSTRAINTS')
                        #print('  comb:', comb_pos, comb_morph)
                        #print('  query:', query_pos, query_morph)
                        #print('  result:', result_pos, result_morph)

                    comb_w1_embedding = get_bert_embeddings(comb[0], tess_data, line_is_global=False, ref=ref)
                    comb_w2_embedding = get_bert_embeddings(comb[1], tess_data, line_is_global=False, ref=ref)
                    comb_embedding = (comb_w1_embedding, comb_w2_embedding)
                    

                    #pairs = tuple(product(row["query_bert"], comb))
                    pairs = tuple(product(query_embeddings, comb_embedding))

                    dists = get_similarities(pairs)
                    dists_sum = compute_bigram_similarity(dists)
                    if not threshold:
                        if dists_sum >= row["bert_similarity"]:
                            results.append((ref, dists_sum, row['bert_query'], list(comb_tokens), comb_pos, comb_morph))
                    else:
                        if dists_sum >= threshold:
                            results.append((ref, dists_sum, row['bert_query'], list(comb_tokens), comb_pos, comb_morph))
    results_.append((row['index'], results))

if print_results:
    # sort results by similarity
    new_results = []
    for result in results_:
        new_results.append(sorted(result[1], key=lambda x: x[1], reverse=True))
    results_ = new_results

    # print results
    for result in results_:
        if len(result) == 0: continue 

        print('Ref:', result[0][0])
        print('  query:', result[0][2])

        for _, dist, _, retrieved, comb_pos, comb_morph in result:
            print(f'  {retrieved} (sim={dist:.4f})')
            print('    pos:', comb_pos)
            print('    morph:', comb_morph[0].get('Gender', ''), comb_morph[1].get('Gender', ''))


# %%
import pickle
import datetime
import os
from statistics import mean

# %%
true_constraints = 'constraints='
if MATCH_TARGET_POS: true_constraints += 'target-pos_'
if MATCH_TARGET_WHOLE_MORPH: true_constraints += 'target-morph_'
if MATCH_QUERY_POS: true_constraints += 'query-pos_'
if MATCH_QUERY_MORPH: true_constraints += 'query-morph_'
if true_constraints.endswith('_'):
    true_constraints = true_constraints[:-1]

if not print_results:
    output_path = f"{os.path.join('temp', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}-results-start-{start_idx}-end-{end_idx}-{true_constraints}-bert.p"
    pickle.dump(results_, open(output_path, 'wb'))
    print(f'Temp results saved at {output_path}')
# For paper results, uncomment to download and run remaining cells
# url = 'https://utexas.box.com/shared/static/2m2e09dijfxgqg3nnxei2cuttc7qu7kd.p'
# urllib.request.urlretrieve (url, 'temp/results_naacl2021_search.p')

# output_path = 'temp/results_naacl2021_search.p'
# results_ = pickle.load(open(output_path, 'rb'))


'''
result_name = '../notebooks/temp/2024-11-24_03-32-36-results.p'
with open(result_name, 'rb') as f:
    results_ = pickle.load(f)
print('Results length:', len(results_))
'''
# Get ranks
ranks = [len(result[1]) for result in results_ if len(result[1]) != 0]

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
