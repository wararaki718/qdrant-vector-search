from typing import List

from qdrant_client.models import ScoredPoint


def show(points: List[ScoredPoint]) -> None:
    for rank, point in enumerate(points, start=1):
        print(f"{rank}-th: id={point.id}, score={point.score}")
    print()


def get_texts() -> List[str]:
    texts = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Python is a versatile programming language used for web development, data analysis, and artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Coding is like solving puzzles; each line of code contributes to the bigger picture.",
        "In a world full of technology, staying curious and learning new things is essential.",
        "Science is organized knowledge. Wisdom is organized life.",
        "Artificial intelligence is reshaping the way we live and work, opening up new possibilities and challenges.",
        "The journey of a thousand miles begins with a single step.",
        "Life is what happens when you're busy making other plans.",
        "The only way to do great work is to love what you do.",
        "Imagination is more important than knowledge. For knowledge is limited, whereas imagination embraces the entire world.",
        "Success is not final, failure is not fatal: It is the courage to continue that counts.",
        "Programming is not about typing, it's about thinking.",
        "The more you know, the more you realize you don't know.",
        "Coding is an art of turning caffeine into code.",
        "The best way to predict the future is to create it.",
        "Keep calm and code on.",
        "Every great developer you know got there by solving problems they were unqualified to solve until they actually did it.",
        "It always seems impossible until it's done.",
    ]
    return texts
