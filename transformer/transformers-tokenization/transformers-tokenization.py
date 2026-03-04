import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        def add_word(word):
            if word in self.word_to_id:
                return
            v = self.vocab_size
            self.word_to_id[word] = v
            self.id_to_word[v] = word
            self.vocab_size += 1

        for word in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            add_word(word)
        
        for text in texts:
            for word in text.split():
                add_word(word)
                
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = []
        for word in text.split():
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                tokens.append(self.word_to_id[self.unk_token])
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join([self.id_to_word[tok] for tok in ids])
