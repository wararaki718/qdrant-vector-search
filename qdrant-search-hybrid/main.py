from qdrant_client.models import (
    Distance,
    NamedVector,
    NamedSparseVector,
    PointStruct,
    SparseIndexParams,
    SearchRequest,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from client import SearchClient
from fusion import reciprocal_rank_fusion
from utils import get_texts, show
from vectorizer import SparseVectorizer, DenseVectorizer


def main():
    collection_name = "sample"
    client = SearchClient()
    dense_params = {
        "text-dense": VectorParams(
            size=768,
            distance=Distance.COSINE,
            on_disk=False,
        ),
    }
    sparse_params = {
        "text-sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        ),
    }
    _ = client.create_index(collection_name, dense_params=dense_params, sparse_params=sparse_params)
    print(f"index created: {collection_name}")

    sparse_model_name = "naver/splade-cocondenser-ensembledistil"
    sparse_vectorizer = SparseVectorizer(sparse_model_name)

    dense_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    dense_vectorizer = DenseVectorizer(dense_model_name)
    
    texts = get_texts()
    points = []
    for point_id, text in enumerate(texts):
        sparse_vector = sparse_vectorizer.transform(text)
        sparse_indices = sparse_vector.nonzero().numpy().flatten()
        sparse_values = sparse_vector.detach().numpy()[sparse_indices]

        dense_vector = dense_vectorizer.transform(text)
        dense_values = dense_vector.detach().numpy()

        point = PointStruct(
            id=point_id,
            payload={},
            vector={
                "text-sparse": SparseVector(
                    indices=sparse_indices.tolist(),
                    values=sparse_values.tolist(),
                ),
                "text-dense": dense_values.tolist(),
            }
        )
        points.append(point)

    client.insert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()

    print("search:")
    top_n = 10
    query_sparse_vector = sparse_vectorizer.transform(texts[0])
    query_sparse_indices = query_sparse_vector.nonzero().numpy().flatten()
    query_sparse_values = query_sparse_vector.detach().numpy()[query_sparse_indices]
    sparse_request = SearchRequest(
        vector=NamedSparseVector(
            name="text-sparse",
            vector=SparseVector(
                indices=query_sparse_indices.tolist(),
                values=query_sparse_values.tolist(),
            ),
        ),
        limit=top_n,
    )

    query_dense_vector = dense_vectorizer.transform(text[0])
    query_dense_values = query_dense_vector.detach().numpy()
    dense_request = SearchRequest(
        vector=NamedVector(
            name="text-dense",
            vector=query_dense_values.tolist(),
        ),
        limit=top_n,
    )

    requests = [sparse_request, dense_request]
    results = client.search(collection_name, requests)
    print("sparse result:")
    show(results[0])

    print("dense result:")
    show(results[1])

    print("hybrid result (reciprocal rank fusion):")
    hybrid_results = reciprocal_rank_fusion(results)
    show(hybrid_results[:top_n])

    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
