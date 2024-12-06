import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

class BERTWrapper:
    def __init__(self, model, tokenizer, w2v_model):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        # Build vocabulary from tokenizer
        self.index_to_key = w2v_model.index_to_key
        self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}

        # Cache for embeddings
        self.embedding_cache = {}
        
    def get_embedding(self, word):
        # Check cache first
        if word in self.embedding_cache:
            return self.embedding_cache[word]
            
        # Compute embedding if not in cache
        inputs = self.tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.hidden_states[-1][0, 1:-1].mean(dim=0)
        
        # Cache the result
        self.embedding_cache[word] = embedding
        return embedding

    def most_similar(self, word, topn=5):
        # Get word embedding
        word_embedding = self.get_embedding(word)

        # Make sure all vocabulary words are in cache
        #for w in self.index_to_key:
        #    if w not in self.embedding_cache:
        #        self.get_embedding(w)
        
        # Convert all embeddings to a single tensor for batch computation
        vocab_embeddings = torch.stack([self.embedding_cache[w] for w in self.index_to_key])
        
        # Compute similarities all at once
        similarities = F.cosine_similarity(word_embedding.unsqueeze(0), vocab_embeddings)
        
        # Get top k indices
        top_k = torch.topk(similarities, k=topn+1)  # +1 because the word itself might be included
        
        # Convert to (word, score) pairs, filtering out the query word
        results = []
        for idx, score in zip(top_k.indices, top_k.values):
            other_word = self.index_to_key[idx]
            if other_word != word:
                results.append((other_word, score.item()))
                if len(results) == topn:
                    break
                    
        return results

    def similarity(self, word1, word2):
        # Get embeddings from cache
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        return F.cosine_similarity(emb1, emb2, dim=0).item()

    def precompute_embeddings(self, word_list=None, batch_size=32, cache_file=None):
        """Precompute embeddings for all words or a specific list"""
        if word_list is None:
            word_list = self.index_to_key
        
        # Load existing cache if available
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            self.embedding_cache = torch.load(cache_file)
        
        # Identify words that still need to be computed
        words_to_compute = [w for w in word_list if w not in self.embedding_cache]
        
        if words_to_compute:
            print(f"Computing embeddings for {len(words_to_compute)} words...")
            # Process in batches
            for i in tqdm(range(0, len(words_to_compute), batch_size)):
                batch_words = words_to_compute[i:i + batch_size]
                # Tokenize all words in batch
                inputs = self.tokenizer(batch_words, padding=True, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Process each word in batch
                for j, word in enumerate(batch_words):
                    # Get the actual token length for this word (excluding padding)
                    word_tokens = len(self.tokenizer.encode(word)) - 2  # subtract 2 for special tokens
                    # Extract embedding for just this word's tokens
                    embedding = outputs.hidden_states[-1][j, 1:1+word_tokens].mean(dim=0)
                    self.embedding_cache[word] = embedding
            
            # Save updated cache if filename provided
            if cache_file:
                print(f"Saving embeddings to {cache_file}")
                torch.save(self.embedding_cache, cache_file)
        else:
            print("All requested words already in cache!")