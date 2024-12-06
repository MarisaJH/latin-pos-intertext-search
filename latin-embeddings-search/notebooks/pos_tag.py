import spacy
import glob
import pickle
from natsort import natsorted
from bert_embeddings import load_data
nlp = spacy.load('la_core_web_lg')
import sys

DEBUG = False
TEXTS_FOLDER = "../data/texts/process"
EMBEDDINGS_FOLDER = "../data/bert_embeddings"
OVERWRITE = False

files = natsorted(glob.glob(f'{TEXTS_FOLDER}/*.tess'))
files_embeddings = natsorted(glob.glob(f'{EMBEDDINGS_FOLDER}/*.pkl'))
# remove any .pos.pkl files
if not OVERWRITE:
    files_embeddings = [f for f in files_embeddings if not f.endswith('.pos.pkl')]

#files = ['../data/bert_embeddings/TESTFILE.txt.pkl']

N = len(files_embeddings)
for i, file in enumerate(files_embeddings):
    print('Processing', file, f'({i}/{N})')
    data = load_data(file)

    # get word tokens
    word_tokens = data['word_tokens']
    pos_tags = []
    morph_tags = []

    
    for line_tokens in word_tokens:
        # if any empty strings, remove them for spacy; but 
        # add empty POS and morph tags for them at the correct index
        empty_indices = [j for j, token in enumerate(line_tokens) if token == '']
        new_line_tokens = [token for token in line_tokens if token != '']
        
        doc = spacy.tokens.Doc(nlp.vocab, words=new_line_tokens)

        this_pos_tags = []
        this_morph_tags = []
        for token in nlp(doc):
            this_pos_tags.append(token.pos_)
            this_morph_tags.append(token.morph.to_dict())
        # add empty POS and morph tags for the empty strings
        for j in empty_indices:
            this_pos_tags.insert(j, '')
            this_morph_tags.insert(j, {})

        pos_tags.append(this_pos_tags)
        morph_tags.append(this_morph_tags)

    # add to data
    data['pos_tags'] = pos_tags
    data['morph_tags'] = morph_tags

    # save
    base_name = file[:-4] # strip .pkl
    save_name = f'{base_name}.pos.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)

    # for debugging, print some info
    if DEBUG:
        print(data['sentences'])
        print(data['word_tokens'])
        print(data['pos_tags'])
        print(data['morph_tags'])

