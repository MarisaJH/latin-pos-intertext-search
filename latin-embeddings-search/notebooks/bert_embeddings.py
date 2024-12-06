from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm import tqdm
import torch
import numpy as np
import re
import pickle
import os
import glob

TEXT_DATA_DIR = "../data/texts/process"
SAVE_DATA_DIR = "../data/bert_embeddings"

def load_text(filepaths: List[str]):
    """Load text and extract line references and content."""
    
    filename_to_references = {}
    filename_to_sentences = {}
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        references = []
        sentences = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Extract reference (e.g., "luc. 1.123") and text
                match = re.match(r'<(.*?)>', line)
                if match:
                    ref = match.group(1)
                    text = line[match.end():].strip()
                    references.append(ref)
                    sentences.append(text)
        
        filename_to_references[filename] = references
        filename_to_sentences[filename] = sentences
    
    return filename_to_references, filename_to_sentences

def get_bert_embeddings(sentences, tokenizer, model):
    """Generate BERT embeddings for sentences."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    embeddings = []
    word_to_tokens_maps = []
    subword_tokens_lists = []
    word_tokens_lists = []
    
    batch_size = 32
    #for i in range(0, len(sentences), batch_size):
    for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences) // batch_size, desc="Generating embeddings"):
        batch = sentences[i:i + batch_size]
        
        encoded = tokenizer(batch, padding=True, truncation=True, 
                          return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        word_ids = [encoded.word_ids(i) for i in range(len(batch))]
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state.cpu().numpy()
            batch_attention_masks = attention_mask.cpu().numpy()
            
            for sent_idx, (sent_emb, sent_mask, sent_word_ids) in enumerate(
                zip(batch_embeddings, batch_attention_masks, word_ids)):
                
                word_to_tokens = {}
                valid_embeddings = sent_emb[sent_mask.astype(bool)]
                valid_tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[sent_idx].tolist()
                )
                
                # Reconstruct words from subword tokens
                current_word = []
                words = []
                for token_idx, (token, word_idx) in enumerate(zip(valid_tokens, sent_word_ids)):
                    if word_idx is not None:  # Skip special tokens
                        if word_idx not in word_to_tokens:
                            word_to_tokens[word_idx] = []
                            if current_word:  # Save previous word if exists
                                words.append(''.join(current_word).replace('Ġ', ''))
                                current_word = []
                        word_to_tokens[word_idx].append(token_idx)
                        current_word.append(token.replace('Ġ', ''))
                
                if current_word:  # Add the last word
                    words.append(''.join(current_word).replace('Ġ', ''))
                
                embeddings.append(valid_embeddings)
                word_to_tokens_maps.append(word_to_tokens)
                subword_tokens_lists.append(valid_tokens)
                word_tokens_lists.append(words)
    
    return embeddings, word_to_tokens_maps, subword_tokens_lists, word_tokens_lists

def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def main():
    model_name = "bowphs/LaBerta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # File paths
    #input_files = [TEXT_DATA_DIR + "/lucan.bellum_civile.part.1.tess"]
    input_files = glob.glob(TEXT_DATA_DIR + "/*.tess")
       
    # Load text
    print("Loading text...")
    filename_to_references, filename_to_sentences = load_text(input_files)
    
    N_FILES = len(filename_to_sentences)
    for filename, sentences in tqdm(filename_to_sentences.items(), total=N_FILES, desc="Processing files"):
        
        # Generate embeddings
        embeddings, word_to_tokens_maps, subword_tokens, word_tokens = get_bert_embeddings(sentences, tokenizer, model)
        
        # Save embeddings and references
        data = {
            'references': filename_to_references[filename],
            'embeddings': embeddings,
            'sentences': sentences,
            'word_to_tokens_maps': word_to_tokens_maps,
            'subword_tokens': subword_tokens,
            'word_tokens': word_tokens
        }
        
        # Create output directory if it doesn't exist
        save_filepath = SAVE_DATA_DIR + "/" + filename + ".pkl"
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        
        # Save to pickle file
        with open(save_filepath, 'wb') as f:
            pickle.dump(data, f)

        # save to txt for debugging
        '''
        with open(save_filepath.replace(".pkl", ".txt"), 'w') as f:
            for i in range(len(data['embeddings'])):
                ref = data['references'][i]
                f.write(f"{ref}\n")
                sent = data['sentences'][i]
                f.write(f"  {sent}\n")
                word_to_tokens = data['word_to_tokens_maps'][i]
                f.write(f"  {word_to_tokens}\n")
                f.write(f"  {data['embeddings'][i]}\n")
                f.write(f"  {data['subword_tokens'][i]}\n")
                f.write(f"  {data['word_tokens'][i]}\n")
                print('embedding shape:', len(data['embeddings'][i]), len(data['embeddings'][i][0]))
        '''
        print(f"Saved {len(embeddings)} embeddings to {save_filepath}")

if __name__ == "__main__":
    main()
    #data = load_data(SAVE_DATA_DIR + "/TESTFILE.txt.pkl")
    #print(data['word_tokens'][0])
    #print(data['subword_tokens'][0])
    #print(data['embeddings'][0])
