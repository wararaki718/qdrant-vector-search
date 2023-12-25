from collections import defaultdict
from typing import Dict, List

from qdrant_client.models import ScoredPoint


def reciprocal_rank_fusion(search_results: List[List[ScoredPoint]], k: int=60) -> List[ScoredPoint]:
    scores = defaultdict(int)
    points: Dict[int, ScoredPoint] = dict()
    for documents in search_results:
        for rank, document in enumerate(documents, start=1):
            scores[document.id] += 1.0 / (k + rank)
            points[document.id] = document
    
    results = []
    for document_id in scores.keys():
        result = points[document_id]
        result.score = scores[document_id]
        results.append(result)
    results.sort(reverse=True, key=lambda result: result.score)

    return results
