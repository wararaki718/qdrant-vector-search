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
    sparse_vectors_config = {
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        ),
    }
    _ = client.create_index(collection_name, sparse_vectors_config=sparse_vectors_config)
    print(f"index created: {collection_name}")

    # load data
    texts = get_texts()

    # insert
    model_name = "naver/splade-cocondenser-ensembledistil"
    vectorizer = SparseVectorizer(model_name)
    points = []
    for point_id, text in enumerate(texts):
        text_vector = vectorizer.transform(text)
        point = PointStruct(id=point_id, payload={}, vector={"sparse": text_vector})
        points.append(point)

    client.insert(collection_name, points)
    print(f"data inserted: {len(points)}")
    print()
    
    # search
    print("search:")
    top_n = 10
    query_vector = vectorizer.transform(texts[0])
    query = NamedSparseVector(name="sparse", vector=query_vector)
    results = client.search(collection_name, query, limit=top_n)
    show(results)

    # delete index
    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
