from transformers import AutoTokenizer, AutoModelForMaskedLM
from bert_utils import BERTWrapper
import re
import numpy as np
from tqdm import tqdm
from pprint import pprint
from gensim.models import Word2Vec

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

# Function for evaluating on Spinelli 2018 synonyms dataset

def syn_eval(model, eval_data, threshold, verbose=False):
    # Better way than two try blocks
    vocab_ = model.key_to_index.keys()
    '''
    try:
        vocab_ = model.vocab.keys()
    except:
        pass
    try: 
        vocab_ = model.wv.vocab.keys()
    except:
        pass
    '''
    with open(eval_data,'r') as f:
        lines = f.readlines()
    total = len(lines)
    matches = 0
    for line in tqdm(lines):
        word, syn = replacer.replace(line.strip()).split('\t')
        if word in vocab_:
            most_sim = [item[0] for item in model.most_similar(word, topn=threshold)]
            most_sim = replacer.replace(" ".join(most_sim)).split()
            if syn in most_sim:
                matches += 1
                if verbose:
                    print(f'Synonym {syn} is in most_similar for {word}')
    return matches/total    

# Function for getting mean reciprocal rank on Spinelli 2018 synonyms dataset

def syn_mrr(model, eval_data, threshold=100, verbose=False):
    # Better way than two try blocks
    vocab_ = model.key_to_index.keys()
    '''
    try:
        vocab_ = model.vocab.keys()
    except:
        pass
    try: 
        vocab_ = model.wv.vocab.keys()
    except:
        pass
    '''
    with open(eval_data,'r') as f:
        lines = f.readlines()
    rrs = []
    for line in tqdm(lines):
        word, syn = replacer.replace(line.strip()).split('\t')
        if word in vocab_ and syn in vocab_:
            most_sim = [item[0] for item in model.most_similar(word, topn=threshold)]
            most_sim = replacer.replace(" ".join(most_sim)).split()
            if syn in most_sim:
                rr = 1 / (most_sim.index(syn) + 1)
                rrs.append(rr)
    mrr = np.mean(rrs)
    return mrr

if __name__ == '__main__':

    eval_path = '../data/evaluationsets'
    syn_eval_data = f'{eval_path}/synonyms.csv' 
    syn_selection_eval_data = f'{eval_path}/syn-selection-benchmark-Latin.tsv'
    replacer = JVReplacer()

    cc100 = Word2Vec.load('../models/latin_w2v_cc100_lemma300_100_1').wv

    # Create a new dictionary with replaced keys
    new_key_to_index = {}
    new_vectors = []
    new_index_to_key = []

    # Iterate through existing keys in order
    for i, key in enumerate(cc100.index_to_key):
        new_key = replacer.replace(key)
        new_key_to_index[new_key] = i
        new_index_to_key.append(new_key)
        new_vectors.append(cc100.vectors[i])

    # Update the model attributes
    cc100.key_to_index = new_key_to_index
    cc100.index_to_key = new_index_to_key
    cc100.vectors = np.array(new_vectors)

    tokenizer = AutoTokenizer.from_pretrained('bowphs/LaBerta')
    bert_model = AutoModelForMaskedLM.from_pretrained('bowphs/LaBerta', output_hidden_states=True)
    laberta = BERTWrapper(bert_model, tokenizer, cc100)

    # Precompute embeddings for evaluation vocabulary
    # First, collect all words that will be used in evaluation
    eval_words = set()
    with open(syn_eval_data, 'r') as f:
        for line in f:
            word, syn = replacer.replace(line.strip()).split('\t')
            eval_words.add(word)
            eval_words.add(syn)

    # Optional: add words from synonym selection dataset if using it
    #try:
    #    with open(syn_selection_eval_data, 'r') as f:
    #        for line in f:
    #            eval_words.update(replacer.replace(line.strip()).split())
    #except FileNotFoundError:
    #    pass

    # Precompute embeddings
    cache_file = "laberta_syn_embeddings_cache.pt"
    laberta.precompute_embeddings(
        #list(eval_words),
        batch_size=32,
        cache_file=cache_file
    )

    # syonym ranking evaluation
    print('syonym ranking evaluation')
    model = laberta
    thresholds = [1,5,10,25,100]
    bert_evals = []

    for threshold in tqdm(thresholds):
        bert_evals.append(syn_eval(laberta, syn_eval_data, threshold))
        
    pprint(list(zip(thresholds, bert_evals)))

    # mrr eval
    print('mrr evaluation')
    bert_mrr = syn_mrr(laberta, syn_eval_data)
    print(bert_mrr)
