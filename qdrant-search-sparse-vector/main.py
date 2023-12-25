from qdrant_client.models import (
    NamedSparseVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
)

from client import SearchClient
from utils import get_texts, show
from vectorizer import SparseVectorizer


def main():
    collection_name = "sample"
    client = SearchClient()

    # create collection
    params = {
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        ),
    }
    _ = client.create_index(collection_name, sparse_params=params)
    print(f"index created: {collection_name}")

    # load data
    texts = get_texts()

    # insert
    model_name = "naver/splade-cocondenser-ensembledistil"
    vectorizer = SparseVectorizer(model_name)
    points = []
    for point_id, text in enumerate(texts):
        text_values, text_indices = vectorizer.transform(text)
        point = PointStruct(
            id=point_id,
            payload={},
            vector={
                "sparse": SparseVector(
                    indices=text_indices,
                    values=text_values,
                )
            }
        )
        points.append(point)

    client.insert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()
    
    # search
    print("search:")
    top_n = 10
    query_values, query_indices = vectorizer.transform(texts[0])
    query = NamedSparseVector(
        name="sparse",
        vector=SparseVector(
            indices=query_indices,
            values=query_values,
        ),
    )
    results = client.search(collection_name, query, limit=top_n)
    show(results)

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
