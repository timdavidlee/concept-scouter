import re
from pathlib import Path

import pandas as pd
from caseconverter import snakecase

abcnumeric = re.compile(r"[A-Za-z0-9- ]")


def load_company_name_data():
    """https://www.kaggle.com/datasets/charanpuvvala/company-classification"""
    file = "/Users/timlee/Documents/data/company-classification/classification-dataset-v1.csv"
    df = pd.read_csv(file)
    df.columns = [snakecase(col) for col in df.columns]
    print("{:,}".format(df.shape[0]))
    return df


def load_product_data():
    """https://www.kaggle.com/datasets/hma2022/amazon-global-store-us-from-saudi-souq"""
    file = "/Users/timlee/Documents/data/amazon-global-store-us/Souq_Saudi_Amazon_Global_Store_US.csv"
    df = pd.read_csv(file)
    df.columns = [snakecase(col) for col in df.columns]
    print("{:,}".format(df.shape[0]))
    return df


def format_kaggle_data():
    extract_dir = Path("sample_data/extracts")
    company_file = extract_dir / "company.tsv"
    item_file = extract_dir / "item_category.tsv"

    company_df = load_company_name_data()
    company_df["category"] = company_df["category"].str.strip()
    company_df["company_name"] = company_df["company_name"].str.strip()

    # reduce to a 2 column file
    company_df = company_df[["category", "company_name"]].drop_duplicates()
    company_df.to_csv(company_file, sep="\t", index=False)
    print("saved: {:,} {}".format(company_df.shape[0], company_file))

    product_data_df = load_product_data()
    product_data_df["item_brand_name"] = product_data_df["item_brand_name"].str.strip()
    product_data_df["item_category_name"] = product_data_df["item_category_name"].str.strip()

    # reduce to a 2 column file
    product_data_df = product_data_df[["item_category_name", "item_brand_name"]].drop_duplicates()
    product_data_df.to_csv(item_file, sep="\t", index=False)
    print("saved: {:,} {}".format(product_data_df.shape[0], item_file))
