"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    _shared_vocab: Dict[str, int] = {}
    _shared_id_to_token: Dict[int, str] = {}

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: int = self.add_token(PAD_TOKEN, 0)
        self.unk_token_id: int = self.add_token(UNK_TOKEN, 1)
        self.bos_token_id: int = self.add_token(BOS_TOKEN, 50256)
        self.eos_token_id: int = self.bos_token_id

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)
        else:
            for token, idx in Vocabulary._shared_vocab.items():
                self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = len(self._token_to_id)
            while idx in self._id_to_token:
                idx += 1
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise ValueError(f"token id must be a non-negative integer, got {idx!r}")
        if idx in self._id_to_token:
            raise ValueError(f"token id {idx} is already assigned to {self._id_to_token[idx]!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        Vocabulary._shared_vocab[token] = idx
        Vocabulary._shared_id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if token in Vocabulary._shared_vocab:
            idx = Vocabulary._shared_vocab[token]
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
            return idx
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        if idx in Vocabulary._shared_id_to_token:
            return Vocabulary._shared_id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return max(self._id_to_token, default=-1) + 1

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id or token in Vocabulary._shared_vocab
