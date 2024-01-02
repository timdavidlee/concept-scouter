"""python -m llama_research.basic_documents.run"""
import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader


def main():
    reader = SimpleDirectoryReader(input_files=[
        "./llama_research/basic_documents/article1.txt",
        "./llama_research/basic_documents/article2.txt",
        "./llama_research/basic_documents/article3.txt",
        "./llama_research/basic_documents/article4.txt",
        "./llama_research/basic_documents/article5.txt",
        "./llama_research/basic_documents/article6.txt",
    ])

    documents = reader.load_data()
    doc1, doc2, doc3, doc4, doc5, doc6 = documents

    doc1.metadata = {
        "company": "Terraform Labs",
        "year": 2024,
    }

    doc2.metadata = {
        "company": "Terraform Labs",
        "year": 2023,
    }

    doc3.metadata = {
        "company": "Grayscale Bitcoin Trust (GBTC)",
        "year": 2023,
    }

    doc4.metadata = {
        "company": "Grayscale Bitcoin Trust (GBTC)",
        "year": 2021,
    }

    doc5.metadata = {
        "game title": "baldurs gate",
        "company": "Larian Game Studios",
        "year": 2024,
    }

    doc6.metadata = {
        "company": "Larian Game Studios",
        "year": 2020,
    }

    index = VectorStoreIndex.from_documents([doc1, doc2, doc3, doc4, doc5, doc6])
    query_engine = index.as_query_engine()

    questions = [
        "What happened to do kwon's company in 2022, and what was it's name?",
        "What happened to do kwon's company in 2023, and what was it's name?",
        "What happened to Barry Silbert's company in 2023, and what was it's name?",
        "What happened to Barry Silbert's company in 2022, and what was it's name?",
        "What happened to Barry Silbert's company in 2021, and what was it's name?",
        "What company is was trading at 50% discount, and what year did it happen?",
        "What happened in the second baldurs gate?",
        "what company made divinity: original sin, and what other game did they make? What was the story about?"
    ]

    for q in questions:
        response = query_engine.query(q)
        print("=" * 80)
        print(q)
        print(response)


if __name__ == "__main__":
    main()
