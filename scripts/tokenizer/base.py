from typing import Dict, List, Tuple
import unicodedata

def get_stats(ids: List[int], stats: Dict=None) -> Dict[int, int]:
    """Get the frequency of each id in the list.

    Args:
        ids (List[int]): List of ids.

    Returns:
        Dict[int, int]: Dictionary of id frequencies.
    """
    stats = {} if stats is None else stats
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats

def merge_stats(ids: List[int], pair: Tuple[int,int], idx:int) -> List[int]:
    """_summary_

    Args:
        ids (List[int]): Ids that is being processed.
        pair (Tuple[int,int]): Pair to be merged.
        idx (int): id of the new merged pair.

    Returns:
        List[int]: Ids after merging the pair.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def replace_control_characters(text: str) -> str:
    """Replace control characters in the text with spaces.

    Args:
        text (str): Input text.

    Returns:
        str: Text with control characters replaced by spaces.
    """
    chars = []
    for ch in text:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(text_as_bytes: bytes) -> str:
    """Render a token as a string, escaping control characters.
    Args:
        text_as_bytes (bytes): Token as bytes.
    Returns:
        str: Token as a string with control characters escaped.
    """
    # pretty print a token, escaping control characters
    s = text_as_bytes.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class Tokenizer:
    """Base class for tokenizers."""

    def __init__(self) -> None:
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        """Train the tokenizer on the given text.
        Args:
            text (str): Text to train on.
            vocab_size (int): Vocabulary size.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        raise NotImplementedError
    
    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token ids.

        Args:
            text (str): Input text.

        Returns:
            List[int]: List of token ids.
        """
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): List of token ids.

        Returns:
            str: Decoded string.
        """
        raise NotImplementedError

    def save(self, save_path: str) -> None:
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        Args:
            save_path (str): Path to save the tokenizer to.
        """
        model_file = save_path + ".model"
        vocab_file = save_path + ".vocab"

        with open(model_file, "wb") as f:
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")

            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file: str) -> 'Tokenizer':
        """Load a tokenizer from a file.

        Args:
            load_path (str): Path to load the tokenizer from.

        Returns:
            Tokenizer: Loaded tokenizer.
        """
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[int, bytes]:
        """Build the vocabulary from special tokens and merges.

        Returns:
            Dict[int, bytes]: Vocabulary mapping token ids to byte sequences.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode('utf-8')
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        return vocab
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary.

        Returns:
            int: Vocabulary size.
        """
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[int, bytes]:
        """Get the vocabulary.

        Returns:
            Dict[int, bytes]: Vocabulary mapping token ids to byte sequences.
        """
        return self.vocab
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get the special tokens.

        Returns:
            Dict[str, int]: Special tokens mapping token strings to ids.
        """
        return self.special_tokens
    
    