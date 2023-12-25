import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


class SparseVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

    def transform(self, text: str) -> torch.Tensor:
        tokens: dict = self._tokenizer(text, return_tensors="pt")
        output = self._model(**tokens)

        weights = torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1)
        vectors, _ = torch.max(weights, dim=1)

        return vectors.squeeze()

    def get_vocabs(self) -> dict:
        return self._tokenizer.get_vocab()



class DenseVectorizer:
    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)

    def transform(self, text: str) -> torch.Tensor:
        inputs: dict = self._tokenizer(text, return_tensors="pt")
        outputs = self._model(**inputs)
        embeddings: torch.Tensor = outputs.last_hidden_state[0].mean(axis=0)
        return embeddings
