import regex as re
from .base import Tokenizer, get_stats, merge_stats
from typing import List, Dict

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class HandleSpecialTokensType:
    NONE_RAISE = "none_raise"   # raise error if special tokens are found
    NONE_IGNORE = "none_ignore" # ignore special tokens (treat as normal text)
    ALL = "all"                 # allow all special tokens

class RegexTokenizer(Tokenizer):
    """A Byte Pair Encoding tokenizer that uses regular expressions to split text into chunks.
       This is similar to the GPT-2 tokenizer, but allows custom regular expressions and special tokens.
    """
    def __init__(self, pattern: str=None):
        super().__init__()
        self.pattern = pattern if pattern is not None else GPT2_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}


    def train(self, text: str, vocab_size: int, verbose=False) -> List[int]:
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        vocab = {i: bytes([i]) for i in range(256)} # int -> bytes
        merges = {} # (int, int) -> int

        for i in range(num_merges):
            # get pair statistics
            
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            # get the most frequent pair
            pair = max(stats, key=stats.get)

            # get a new index for the merged pair
            idx = 256 + i

            # merge the most frequent pair
            ids = [merge_stats(id_seq, pair, idx) for id_seq in ids]

            # update the vocab and merges
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        self.merges = merges
        self.vocab = vocab
    
    def register_special_tokens(self, special_tokens: Dict[str, int]):
        """Register special tokens.
        Args:
            special_tokens (Dict[str, int]): A dictionary of special tokens.
                The keys are the token strings, and the values are their corresponding ids.
                The ids should be >= 256 to avoid conflict with byte tokens.
        """
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids: List[bytes]) -> str:
        """Decode a list of token ids back into a string.
        Args:
            ids (List[int]): List of token ids.
        Returns:
            str: Decoded string.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text: str) -> List[int]:
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

    def _encode_ignore_special(self, text: str) -> List[int]:
        """Encode text into a list of token ids, ignoring special tokens.
        Args:
            text (str): Input text.
        Returns:
            List[int]: List of token ids.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for ch in text_chunks:
            ids.extend(self._encode_chunk(ch))
        return ids
    
    def encode(self, text: str, allowed_special: HandleSpecialTokensType= HandleSpecialTokensType.NONE_RAISE) -> List[int]:
        """Encode text into a list of token ids, handling special tokens.
        Args:
            text (str): Input text.
        Returns:
            List[int]: List of token ids.
        """
        # handle special tokens
        special = {}
        if allowed_special == HandleSpecialTokensType.ALL:
            special = self.special_tokens
            
        elif allowed_special == HandleSpecialTokensType.NONE_IGNORE:
            print("Warning: ignoring special tokens in the input text")

        elif allowed_special == HandleSpecialTokensType.NONE_RAISE:
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            return self._encode_ignore_special(text)
        
        # split by special tokens
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids