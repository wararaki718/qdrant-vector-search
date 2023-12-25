import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    docid: int
    score: float

    def __lt__(self, other: "Document") -> bool:
        return self.score < other.score


def show(documents: List[Document]) -> None:
    for i, document in enumerate(documents):
        print(f"{i}: docid={document.docid}, score={document.score}")
    print()


def get_documents(n: int) -> List[Document]:
    documents = []
    for i in range(1, n+1):
        documents.append(Document(
            docid=i,
            score=random.random(),
        ))
    documents.sort(reverse=True)
    return documents


def reciprocal_rank_fusion(documents1: List[Document], documents2: List[Document], k: int=60) -> List[Document]:
    scores = defaultdict(int)
    for rank, document in enumerate(documents1, start=1):
        scores[document.docid] += 1.0 / (k + rank)
    
    for rank, document in enumerate(documents2, start=1):
        scores[document.docid] += 1.0 / (k + rank)
    
    results = [Document(docid=key, score=value) for key, value in scores.items()]
    results.sort(reverse=True)

    return results
        

def main() -> None:
    n = 10
    documents1 = get_documents(n)
    documents2 = get_documents(n)

    print("document1 ranking:")
    show(documents1)
    print("document2 ranking:")
    show(documents2)

    print("reciprocal rank fusion:")
    results = reciprocal_rank_fusion(documents1, documents2, k=1)
    show(results)

    print("DONE")


if __name__ == "__main__":
    main()
