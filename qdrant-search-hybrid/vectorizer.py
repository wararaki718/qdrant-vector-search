from typing import List, Tuple

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


class DenseVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

    def transform(self, text: str) -> List[float]:
        inputs: dict = self._tokenizer(text, return_tensors="pt")
        outputs = self._model(**inputs)
        embeddings: torch.Tensor = outputs.last_hidden_state[0].mean(axis=0)
        return embeddings.cpu().detach().tolist()


class SparseVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

    def transform(self, text: str) -> Tuple[List[float], List[float]]:
        tokens: dict = self._tokenizer(text, return_tensors="pt")
        output = self._model(**tokens)

        weights = torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1)
        embeddings, _ = torch.max(weights, dim=1)
        embeddings = embeddings.squeeze()
        
        indices = embeddings.nonzero().squeeze().tolist()
        vectors = embeddings[indices].tolist()

        return vectors, indices

    def get_vocabs(self) -> dict:
        return self._tokenizer.get_vocab()
