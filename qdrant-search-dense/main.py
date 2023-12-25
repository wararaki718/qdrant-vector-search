from qdrant_client.models import (
    Distance,
    NamedVector,
    PointStruct,
    SearchRequest,
    VectorParams,
)

from client import SearchClient
from utils import get_texts, show
from vectorizer import DenseVectorizer


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
    _ = client.create_index(collection_name, dense_params=dense_params)
    print(f"index created: {collection_name}")

    dense_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    dense_vectorizer = DenseVectorizer(dense_model_name)
    
    texts = get_texts()
    points = []
    for point_id, text in enumerate(texts):
        dense_vector = dense_vectorizer.transform(text)
        dense_values = dense_vector.detach().numpy()

        point = PointStruct(
            id=point_id,
            payload={},
            vector={
                "text-dense": dense_values.tolist(),
            }
        )
        points.append(point)

    client.insert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()

    print("search:")
    top_n = 10

    query_dense_vector = dense_vectorizer.transform(text[0])
    query_dense_values = query_dense_vector.detach().numpy()
    dense_request = SearchRequest(
        vector=NamedVector(
            name="text-dense",
            vector=query_dense_values.tolist(),
        ),
        limit=top_n,
    )

    requests = [dense_request]
    results = client.search(collection_name, requests)

    print("dense result:")
    show(results[0])

    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
