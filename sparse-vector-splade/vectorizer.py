from typing import Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class TextVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

    def transform(self, text: str) -> Tuple[torch.Tensor, dict]:
        tokens: dict = self._tokenizer(text, return_tensors="pt")
        output = self._model(**tokens)

        weights = torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1)
        vectors, _ = torch.max(weights, dim=1)

        return vectors.squeeze(), tokens

    def get_vocabs(self) -> dict:
        return self._tokenizer.get_vocab()
