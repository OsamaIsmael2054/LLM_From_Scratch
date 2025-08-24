from .base import Tokenizer, merge_stats, get_stats
from typing import List

class BasicTokenizer(Tokenizer):
    """Minimal (byte-level) Byte Pair Encoding tokenizer.
       Without any regular expression or special tokens.
    """

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False) -> List[int]:
        """Tokenize the input text into a list of token ids.

        Args:
            text (str): Input text.

        Returns:
            List[int]: List of token ids.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"

        num_merges = vocab_size - 256
        text_as_bytes = text.encode('utf-8', errors='replace')
        ids = list(text_as_bytes) # list of ints (bytes)

        vocab = {i: bytes([i]) for i in range(256)} # int -> bytes
        merges = {} # (int, int) -> int

        for i in range(num_merges):
            # get pair statistics
            stats = get_stats(ids)

            # get the most frequent pair
            pair = max(stats, key=stats.get)

            # get a new index for the merged pair
            idx = 256 + i

            # merge the most frequent pair
            ids = merge_stats(ids, pair, idx)

            # update the vocab and merges
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def decode(self, token_ids):
        """Decode a list of token ids back into a string.

        Args:
            token_ids (List[int]): List of token ids.

        Returns:
            str: Decoded string.
        """
        bytes_list = [self.vocab[token_id] for token_id in token_ids]
        text_as_bytes = b''.join(bytes_list)
        text = text_as_bytes.decode('utf-8', errors='replace')
        return text
    
    def encode(self, text):
        """Tokenize the input text into a list of token ids.

        Args:
            text (str): Input text.

        Returns:
            List[int]: List of token ids.
        """
        text_as_bytes = text.encode('utf-8', errors='replace')
        ids = list(text_as_bytes)
        while len(ids) >= 2:
            # find the first pair that can be merged
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge_stats(ids, pair, idx)
        return ids
       