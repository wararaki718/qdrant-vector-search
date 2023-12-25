import json

from postprocessor import Postprocessor
from vectorizer import TextVectorizer


def main() -> None:
    model_name = "naver/splade-cocondenser-ensembledistil"
    vectorizer = TextVectorizer(model_name)

    text = "Tamagoyaki is one of the egg dishes"
    vectors, tokens = vectorizer.transform(text)
    print(vectors.shape)

    postprocessor = Postprocessor(vectorizer.get_vocabs())
    result = postprocessor.transform(vectors)
    print(json.dumps(result, indent=4))
    print("DONE")


if __name__ == "__main__":
    main()
    