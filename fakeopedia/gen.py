"""python -m fakeopedia.gen"""
import pandas as pd

from loguru import logger
from fakeopedia.trxn_generation.data import load_company_info, load_item_catalog
from fakeopedia.trxn_generation.trxn_factory import TransactionFactory


def main():
    company_df = load_company_info(cache_flag=False)
    catalog_df = load_item_catalog(cache_flag=False)
    cat2iid = catalog_df.groupby("item_category_name")["iid"].agg(list)
    fact = TransactionFactory(cat2iid, sample_company_size=500)
    fact.generate_transactions()

    txn_df = pd.DataFrame(
        fact.transactions,
        columns=["company_id", "item_category", "iid", "country", "purchase_date"]
    )

    txn_df["purchase_month"] = txn_df["purchase_date"].dt.month

    merged_df = txn_df.merge(
        company_df,
        how="left",
        on="company_id"
    ).merge(
        catalog_df[["iid", "item_brand_name", "item_price_usd", "item_title"]],
        how="left",
        on="iid"
    )

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