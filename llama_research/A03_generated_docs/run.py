"""python -m llama_research.A03_generated_docs.run"""


import pandas as pd
import numpy as np

from loguru import logger
from llama_index import Document as LlamaDoc, VectorStoreIndex
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo

np_rand = np.random.RandomState(997)


def load_some_data(size: int = 3):
    raw_df = pd.read_feather("/tmp/trxn.feather")
    company_names = np_rand.choice(raw_df["company_name"].unique(), size=size)
    mask = raw_df["company_name"].isin(company_names)
    df = raw_df[mask].copy()
    return df

def agg_the_df(df: pd.DataFrame, agg_dim: str):
    agg_df = df.groupby(agg_dim).agg(
        sales_count=pd.NamedAgg("amt", "size"),
        sales_amount=pd.NamedAgg("amt", "sum"),
    ).sort_values("sales_amount", ascending=False)
    agg_df["sales_amount"] = agg_df["sales_amount"].round(2)
    agg_df["rank by amount"] = agg_df["sales_amount"].rank(method="dense", ascending=False) 
    agg_df["rank by count"] = agg_df["sales_count"].rank(method="dense", ascending=False)
    return agg_df


def create_text_doc(title: str, body: str, subtitle: str = ""):
    full_body_text = (
        f"{title}\n"
        f"{subtitle}\n"
        f"{body}"
    )

    return full_body_text


dims = [
    "company",
    "year",
    "country",
    "item category",
    "brand",
]


def write_doc_to_repo(filename: str, doctext: str):
    fullpath = "./llama_research/A03_generated_docs/generated/{}".format(filename) 
    with open(fullpath, "w") as f:
        f.write(doctext)
    logger.info("saved {} chars to: {}".format(len(doctext), fullpath))


def generate_docs():
    df = load_some_data(size=15)
    df = df.rename(columns={
        "company_name": "company",
        "purchase_year": "year",
        "item_category": "item category",
        "item_brand_name": "brand",
    })

    df["amt"] = df["item_price_usd"]

    doc_collector = []
    for doc_dim in dims:
        if doc_dim == "item_brand_name":
            continue

        rem_dims = [d for d in dims if d != doc_dim]
        unique_vals = df[doc_dim].unique()
        high_level_df = agg_the_df(df, doc_dim)
        doctext = create_text_doc(
            title=f"== Title: Sales Figure Summary for {doc_dim} ==",
            subtitle=f"The following covers sales statistics for all {rem_dims}",
            body=high_level_df.to_markdown(floatfmt=".0f")
        )
        doc_id = f"{doc_dim}-Summary"
        write_doc_to_repo(doc_id + ".txt", doctext)
        llmdoc = LlamaDoc(doc_id=doc_id, text=doctext)
        logger.info(f"created doc_id: {doc_id}")
        doc_collector.append(llmdoc)

        for uv in unique_vals:
            title = f"== Title: Sales Figures for {doc_dim}: {uv} == \n\n"            
            mask = df[doc_dim] == uv
            subset_df = df[mask]

            bodytext = ""
            for agg_dim in rem_dims:
                bodytext += f"== {uv}: Sales Break out by {agg_dim} == \n\n"
                agg_df = agg_the_df(subset_df, agg_dim)
                bodytext += agg_df.to_markdown(floatfmt=".0f") + "\n\n"

            doctext = create_text_doc(title=title, body=bodytext)
            doc_id = f"{doc_dim}-{uv}"
            write_doc_to_repo(doc_id + ".txt", doctext)

            llmdoc = LlamaDoc(doc_id=doc_id, text=doctext)
            logger.info(f"created doc_id: {doc_id}")
            doc_collector.append(llmdoc)


if __name__ == "__main__":
    generate_docs()
