"""python -m fakeopedia.gen"""
import os
import pandas as pd

from loguru import logger
from fakeopedia.trxn_generation.data import load_company_info, load_item_catalog
from fakeopedia.trxn_generation.trxn_factory import TransactionFactory


def load_trxn(
    cache_flag: bool = True,
    n_companies: int = 500,
    n_trxn_per: int = 10_000
):
    cache_file = "/tmp/trxn.feather"
    if os.path.exists(cache_file) and cache_flag:
        logger.info("cached version found: {}".format(cache_file))
        return pd.read_feather(cache_file)

    company_df = load_company_info(cache_flag=False)
    catalog_df = load_item_catalog(cache_flag=False)
    cat2iid = catalog_df.groupby("item_category_name")["iid"].agg(list)

    fact = TransactionFactory(
        cat2iid,
        n_transactions=n_trxn_per,
        sample_company_size=n_companies
    )
    fact.generate_transactions()

    txn_df = pd.DataFrame(
        fact.transactions,
        columns=["company_id", "item_category", "iid", "country", "purchase_date"]
    )

    txn_df["purchase_month"] = txn_df["purchase_date"].dt.strftime("%Y-%m")

    merged_df = txn_df.merge(
        company_df,
        how="left",
        on="company_id"
    ).merge(
        catalog_df[["iid", "item_brand_name", "item_price_usd", "item_title"]],
        how="left",
        on="iid"
    )

    merged_df.to_feather(cache_file)
    logger.info("saved cached file to: {}".format(cache_file))
    return merged_df


def main(cache_flag: bool = True):
    merged_df = load_trxn(cache_flag)

    dims = [
        "company_category",
        "company_name",
        "item_category",
        "item_brand_name",
        "country",
        "purchase_month"
    ]
    logger.info(merged_df.head(10).to_markdown())
    for d in dims:
        logger.info("=" * 80)
        logger.info("\n" + merged_df[d].value_counts().to_markdown())


if __name__ == "__main__":
    main()
