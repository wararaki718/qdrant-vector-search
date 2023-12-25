from qdrant_client.models import SparseVectorParams, SparseIndexParams, PointStruct, SparseVector, NamedSparseVector

from client import SearchClient
from utils import get_texts, show
from vectorizer import TextVectorizer


def main():
    collection_name = "sample"
    client = SearchClient()
    params = {
        "text": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        ),
    }
    _ = client.create_index(collection_name, sparse_params=params)
    print(f"index created: {collection_name}")


    model_name = "naver/splade-cocondenser-ensembledistil"
    vectorizer = TextVectorizer(model_name)
    
    texts = get_texts()
    points = []
    for point_id, text in enumerate(texts):
        text_vector = vectorizer.transform(text)
        text_indices = text_vector.nonzero().numpy().flatten()
        text_values = text_vector.detach().numpy()[text_indices]
        point = PointStruct(
            id=point_id,
            payload={},
            vector={
                "text": SparseVector(
                    indices=text_indices.tolist(),
                    values=text_values.tolist(),
                )
            }
        )
        points.append(point)
    print(f"data inserted: {len(points)}")
    
    client.insert(collection_name, points)

    query_vector = vectorizer.transform(texts[0])
    query_indices = query_vector.nonzero().numpy().flatten()
    query_values = query_vector.detach().numpy()[query_indices]
    query = NamedSparseVector(
        name="text",
        vector=SparseVector(
            indices=query_indices.tolist(),
            values=query_values.tolist(),
        ),
    )
    result = client.search(collection_name, query)
    show(result)

    _ = client.delete_index(collection_name)
    print(f"index deleted: {collection_name}")

    print("DONE")


if __name__ == "__main__":
    main()
