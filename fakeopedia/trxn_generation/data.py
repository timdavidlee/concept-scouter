import os
import locale

import pandas as pd
from loguru import logger
from caseconverter import snakecase

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def load_item_catalog(cache_flag: bool = True):
    cache_file = "/tmp/catalog.feather"
    if cache_flag and os.path.exists(cache_file):
        logger.info(f"found {cache_file}")
        return pd.read_feather(cache_file)

    src_file = "/Users/timlee/Documents/data/amazon-global-store-us/Souq_Saudi_Amazon_Global_Store_US.csv"

    df = pd.read_csv(src_file)
    df.columns = [snakecase(col) for col in df.columns]

    df["item_price_usd"] = df["item_price"].map(lambda x: locale.atof(x)) * 0.27
    df["item_price_usd"] = df["item_price_usd"].round(2)
    df["iid"] = df.index.values
    df.to_feather(cache_file)
    logger.info(f"saved: {cache_file}")
    return df


def load_company_info(cache_flag: bool = True):
    cache_file = "/tmp/company.feather"
    if cache_flag and os.path.exists(cache_file):
        logger.info(f"found {cache_file}")
        return pd.read_feather(cache_file)

    src_file = "/Users/timlee/Documents/data/company-classification/classification-dataset-v1.csv"
    df = pd.read_csv(src_file)
    df.columns = [snakecase(col) for col in df.columns]
    df = df[["category", "company_name"]]
    df["company_name"] = df["company_name"].map(lambda x: x.title())

    df = df.reset_index().rename(columns={
        "index": "company_id",
        "category": "company_category"
    })

    df.to_feather(cache_file)
    logger.info(f"saved: {cache_file}")
    return df
