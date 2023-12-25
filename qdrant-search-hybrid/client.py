from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, ScoredPoint, UpdateResult, SparseVectorParams, SearchRequest, VectorParams


class SearchClient:
    def __init__(self, host: str="localhost", port: int=6333):
        self._client = QdrantClient(host=host, port=port)

    def create_index(self, collection_name: str, dense_params: Dict[str, VectorParams], sparse_params: Dict[str, SparseVectorParams]) -> bool:
        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=dense_params,
            sparse_vectors_config=sparse_params,
        )
    
    def insert(self, collection_name: str, points: List[PointStruct]) -> UpdateResult:
        response = self._client.upsert(
            collection_name=collection_name,
            points=points
        )
        return response

    def search(self, collection_name: str, requests: List[SearchRequest]) -> List[List[ScoredPoint]]:
        response = self._client.search_batch(
            collection_name=collection_name,
            requests=requests,
        )
        return response

    def delete_index(self, collection_name: str) -> bool:
        response = self._client.delete_collection(collection_name=collection_name)
        return response
