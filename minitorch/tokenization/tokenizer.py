from collections import Counter
from typing import List, Dict, Tuple

class Tokenizer:
    """
    Base Tokenizer providing the basic architecture that tokenizers follow.
    
    It provides the following architecture
    - encode(): convert text to token ID.
    - decode(): convert token IDs back to text.
    """
    
    #* predefined tokens
    TOKEN_UNKNOWN= "<UNK>"
    TOKEN_EOT = "<EOT>"
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            List[int]: A list of token IDs corresponding to the input text.
        """
        raise NotImplementedError(
            f"encode() not implemented on Tokenizer class.\n"
            f" ❌ Called encode() on abstract base class {self.__class__.__name__}\n"
            f" 🔦 Tokenizer class is just an interface. Implement encode() in concrete class such as CharTokenizer() or BPETokennizer().\n"
            f" ✅ Example: tokenizer = CharTokenizer(); token_ids = tokenizer.encode('hello world')")
        
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token ids back to raw text

        Args:
            token_ids (List[int]): IDs to convert to text

        Returns:
            str: A string of text obtained from the token Ids
        """
        raise NotImplementedError(
            f"decode() not implemented on Tokenizer class.\n"
            f" ❌ Called decode() on abstract base class {self.__class__.__name__}\n"
            f" 🔦 Tokenizer class is just an interface. Implement decode() in concrete class such as CharTokenizer() or BPETokennizer().\n"
            f" ✅ Example: tokenizer = CharTokenizer(); text = tokenizer.decode([1, 2, 3])")
    

class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that converts text into token IDs based on individual characters.
    
    This tokenizer creates a vocabulary of unique characters from the input text and
    assigns a unique ID to each character. It also includes special tokens for unknown
    characters and end-of-text.
    
    Character tokenization provides a simple, robust foundation for text processing. 
    The key insight is that with a small vocabulary (typically <100 characters),
    we can represent any text without unknown tokens.

    **Trade-offs**:
    - **Pro**: No out-of-vocabulary issues, handles any language
    - **Con**: Long sequences (1 char = 1 token), limited semantic understanding
    - **Use case**: When robustness is more important than efficiency
    """
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.next_id: int = 0
        
        # Add special tokens to the vocabulary
        self._add_token(self.TOKEN_UNKNOWN)
        self._add_token(self.TOKEN_EOT)
        
    def _add_token(self, token: str) -> None:
        """
        Add a token to the vocabulary and return its assigned ID.
        
        Args:
            token (str): The token to be added to the vocabulary.
        """
        if token not in self.char_to_id:
            self.char_to_id[token] = self.next_id
            self.id_to_char[self.next_id] = token
            self.next_id += 1
            
        self.vocab_size: int = len(self.char_to_id)
    
    def build_vocab(self, corpus: List[str]) -> None:
        """
        Build the vocabulary from a list of text samples.
        
        Args:
            corpus (List[str]): A list of text samples to build the vocabulary from.
        """
        #* lower case and strip whitespace from each text sample
        corpus = [text.lower().strip() for text in corpus]
        
        #* Iterate through each text sample and add each character to the vocabulary
        for text in corpus:
            for char in text:
                self._add_token(char)
                
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs based on the character-level vocabulary.
        
        Args:
            text (str): The input text to be tokenized.
        
        Returns:
            List[int]: A list of token IDs corresponding to the input text.
        """
        token_ids = []
        for char in text:
            token_id = self.char_to_id.get(char, self.char_to_id[self.TOKEN_UNKNOWN])
            token_ids.append(token_id)
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to raw text based on the character-level vocabulary.
        
        Args:
            token_ids (List[int]): A list of token IDs to be converted back to text.
        
        Returns:
            str: A string of text obtained from the token IDs.
        """
        chars = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, 0)
            chars.append(char)
        return ''.join(chars)