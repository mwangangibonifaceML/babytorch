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
        Add a token to the vocabulary.
        
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
    
class BPETokenizer(Tokenizer):

    """
    Byte Pair Encoding (BPE) that learns subwords units.
    
    Starts by with character level vocabulary then learns
    the most frequent character pairs and merges them together
    to form single tokens.
    
    Keeps repeating until desired vocabulary size is attained.

    """
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size:     int = vocab_size
        self.vocab:          List = []
        self.tokens_to_ids:  Dict[str, int] = {}
        self.ids_to_tokens:  Dict[int, str] = {}
        self.merges:         List[Tuple[str, str]] = []
        
    def pre_tokenize(self, text: str) -> List[str]:
        """
        Splits text into words and punctuation while preserving 
        essential boundaries.
        """
        return re.findall(r'\w+|[^\w\s]', text)
        
    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Get the individual tokens of a word

        Args:
            word (str): The word to get tokens from

        Returns:
            List[str]: Individual tokens from a word
        """
        tokens = list(word)
        tokens[-1] += Tokenizer.TOKEN_EOT
        return tokens
    
    
    def train(self, corpus: List[str], vocab_size: int):
        """
        Train the BPE to learn pairs and merge them.
        
        It initializes character vocabulary and run a greedy merge loop
        using _count_byte_pairs to find the best pair and _merge_pairs()
        to merge them

        Args:
            corpus (List[str]): Document to learn and train from (expected to be a list of sentences)
            vocab_size (int): maximum vocabulary size allowed
        """
        full_corpus = []
        
        if corpus is None or len(corpus) == 0:
            raise ValueError("Corpus cannot be empty. Please provide a valid corpus for training.")
        
        for sentence in corpus:
            full_corpus.extend(self.pre_tokenize(sentence))

        if vocab_size:
            self.vocab_size = vocab_size
            
        #* count word occurences in the corpus to get the frequency of each word
        word_freq: Counter = Counter(full_corpus)
        word_tokens: Dict[str, list[str]] = {}
        vocab = set()
        
        #* get the words tokens from corpus
        for word in full_corpus:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)
        
        #* update the vocabulary using the tokens and
        #* unknown token    
        self.vocab = sorted(list(vocab))
        
        if Tokenizer.TOKEN_UNKNOWN not in self.vocab:
            self.vocab.insert(0, Tokenizer.TOKEN_UNKNOWN)
            
        #* find the best pair(s) and merge them
        while len(self.vocab) < self.vocab_size:
            pair_counts = self._count_byte_pairs(word_tokens, word_freq)
            if not pair_counts:
                break
            
            best_pair = pair_counts.most_common(1)[0][0]
            new_token = self._merge_pair(word_tokens, best_pair)
    
            self.merges.append(best_pair)
            self.vocab.append(new_token)
            
        self._build_mapping()
        
    def _build_mapping(self):
        """
        Create token to id and id to token mappings
        """
        self.tokens_to_ids = {token: id for id, token in enumerate(self.vocab)}
        self.ids_to_tokens = {id: token for id, token in enumerate(self.vocab)}
            
    def _count_byte_pairs(self, word_token: Dict[str, List[str]], word_count: Counter)-> Counter:
        """
        Count the frequency of all adjacent token pairs in a corpus
        
        Each pair count is weighted by the frequency of word containing it
        in the corpus, so most frequent words contribute more the statistic

        Args:
            word_token (Dict[str, List[str]]): Dictionary with word and its tokens
            word_count (Counter): Dictionary showing the frequency count of the word

        Returns:
            Counter: Pair count freqeuncies of the adjacent tokens
        """
        pair_count = Counter()
        for word, count in word_count.items():
            tokens = word_token[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_count[pair] += count
                
        return pair_count
                
    def _merge_pair(self, word_token: Dict[str, List[str]], pair: Tuple[str, str]) -> str:
        """
        Merge the most frequent pair in all word token lists.

        Scans through every word's tokens and replaces adjacent occurrences
        of the pair with a single concatenated token. Modifies word_tokens
        in place and returns the new merged token string.
        """
        merged_pair = pair[0] + pair[1]
        
        for word in word_token:
            tokens = word_token[word]
            new_tokens: List[str] = []
            counter = 0
            
            
            while counter < len(tokens):
                if (counter < len(tokens) - 1) and tokens[counter] == pair[0]\
                    and tokens[counter + 1] == pair[1]:
                    new_tokens.append(merged_pair)
                    counter += 2
                else:
                    new_tokens.append(tokens[counter])
                    counter += 1
            word_token[word] = new_tokens
                    
        return merged_pair
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        if not self.merges:
            return tokens
        
        current_tokens = list(tokens)
        for pair in self.merges:
            new_tokens = []
            i = 0
            
            while (i < len(current_tokens)):
                if (i < len(current_tokens) - 1) and\
                current_tokens[i] == pair[0] and \
                current_tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
                    
            current_tokens = new_tokens
        return current_tokens
    
    def encode(self, text: str)-> List[int]:
        words = text.split()
        all_tokens = []
        
        for word in words:
            tokens = self._get_word_tokens(word)
            merged_tokens = self._apply_merges(tokens)
            all_tokens.extend(merged_tokens)
        
        tokens_ids = []
        for token in all_tokens:
            id = self.tokens_to_ids.get(token, 0)
            tokens_ids.append(id)
        
        return tokens_ids
    
    def decode(self, token_ids: List[int]) -> str:
        #* return empty string id token mapping doesn't exist
        if not self.ids_to_tokens:
            return ""
        
        #* iterate through the token ids and get the corresponding
        #* token from id to token mapping
        text = [self.ids_to_tokens.get(t, Tokenizer.TOKEN_UNKNOWN) for t in token_ids]
        
        #* join all the tokens together and perform some clean up
        text = "".join(text)
        text = text.replace(Tokenizer.TOKEN_EOT, " ")
        text = " ".join(text.split())
        return text