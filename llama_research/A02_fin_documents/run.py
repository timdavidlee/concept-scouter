import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader


def write_article_to_disk(long_str: str, filename: str):
    with open(filename, "w") as f:
        f.write(long_str)


def main():
    article1 = """
    Top Sellers
    Pencils, 500 sales, 50 million USD
    Tables, 400 sales, 40 million USD
    Chairs, 300 sales, 30 million USD
    """

    article2 = """
    Top Sellers
    Tables, 600 sales, 60 million USD
    Paper, 200 sales, 20 million USD
    Knives, 100 sales, 10 million USD
    """

    article3 = """
    Top Sellers
    Pens, 1000 sales, 100 million USD
    Tables, 800 sales, 80 million USD
    Towels, 30 sales, 3 million USD
    """

    article4 = """
    Top Sellers
    Blankets, 700 sales, 70 million USD
    Quilts, 100 sales, 10 million USD
    Snacks, 70 sales, 7 million USD
    """

    write_article_to_disk(article1, "/tmp/article1")
    write_article_to_disk(article2, "/tmp/article2")
    write_article_to_disk(article3, "/tmp/article3")
    write_article_to_disk(article4, "/tmp/article4")


    reader = SimpleDirectoryReader(input_files=["/tmp/article1", "/tmp/article2", "/tmp/article3", "/tmp/article4", "/tmp/article5", "/tmp/article6"])
    documents = reader.load_data()
    doc1, doc2, doc3, doc4, doc5, doc6 = documents

    doc1.metadata = {
        "region": "US",
        "year": 2024,
    }

    doc2.metadata = {
        "region": "US",
        "year": 2023,
    }

    doc3.metadata = {
        "region": "EU",
        "year": 2024,
    }

    doc4.metadata = {
        "region": "EU",
        "year": 2023,
    }


    index = VectorStoreIndex.from_documents([doc1, doc2, doc3, doc4])
    query_engine = index.as_query_engine()

    questions = [
        "In which region had snacks as the top seller?",
        "What year was pencils a top seller?",
        "Which years and which regions was Tables a top seller?",
        "From all the years where tables was the top seller, what was largest amount?",
        "Was snacks ever a top seller in the US?",
        "Show every year and the sales amount for tables",
    ]

    for q in questions:
        response = query_engine.query(q)
        print("=" * 80)
        print(q)
        print(response)

