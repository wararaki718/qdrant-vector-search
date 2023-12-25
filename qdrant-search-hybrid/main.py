from qdrant_client.models import (
    Distance,
    NamedVector,
    NamedSparseVector,
    PointStruct,
    SparseIndexParams,
    SearchParams,
    SearchRequest,
    SparseVectorParams,
    VectorParams,
)

from client import SearchClient
from fusion import reciprocal_rank_fusion
from utils import get_texts, show
from vectorizer import DenseVectorizer, SparseVectorizer


def main():
    collection_name = "sample"
    client = SearchClient()

    # create collection
    dense_vectors_config = {
        "dense": VectorParams(
            size=768,
            distance=Distance.COSINE,
            on_disk=False,
        ),
    }
    sparse_vectors_config = {
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False),
        ),
    }
    _ = client.create_index(
        collection_name,
        vectors_config=dense_vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"index created: {collection_name}")

    # load data
    texts = get_texts()

    # index
    dense_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    dense_vectorizer = DenseVectorizer(dense_model_name)

    sparse_model_name = "naver/splade-cocondenser-ensembledistil"
    sparse_vectorizer = SparseVectorizer(sparse_model_name)
    points = []
    for point_id, text in enumerate(texts):
        dense_vector = dense_vectorizer.transform(text)
        sparse_vector = sparse_vectorizer.transform(text)
        
        point = PointStruct(
            id=point_id,
            payload={},
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
            }
        )
        points.append(point)

    client.insert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()

    # search
    print("search:")
    top_n = 10
    query_dense_vector = dense_vectorizer.transform(text[0])
    dense_request = SearchRequest(
        vector=NamedVector(
            name="dense",
            vector=query_dense_vector,
        ),
        limit=top_n,
    )

    query_sparse_vector = sparse_vectorizer.transform(texts[0])
    sparse_request = SearchRequest(
        vector=NamedSparseVector(
            name="sparse",
            vector=query_sparse_vector,
        ),
        params=SearchParams(hnsw_ef=128, exact=False),
        limit=top_n,
    )

    requests = [dense_request, sparse_request]
    results = client.search(collection_name, requests)
    print("dense result:")
    show(results[0])

    print("sparse result:")
    show(results[1])

    print("hybrid result (reciprocal rank fusion):")
    hybrid_results = reciprocal_rank_fusion(results)
    show(hybrid_results[:top_n])

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
