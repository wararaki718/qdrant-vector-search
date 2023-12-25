from typing import Dict

import torch


class Postprocessor:
    def __init__(self, vocabs: Dict[str, int]) -> None:
        self._index2token = dict(zip(vocabs.values(), vocabs.keys()))

    def transform(self, vectors: torch.Tensor) -> Dict[str, float]:
        indices = vectors.nonzero().squeeze().cpu().tolist()
        weights = vectors[indices].cpu().tolist()

        result = {
            self._index2token.get(index): weight for index, weight in zip(indices, weights)
        }
        return result
