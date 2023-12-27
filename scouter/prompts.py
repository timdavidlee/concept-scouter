"""python -m scouter.prompts"""
import os
import json
import numpy as np
import pandas as pd
from hashlib import md5

from openai import OpenAI

COMPANY_DATA = "./sample_data/extracts/company.tsv"
ITEM_DATA = "./sample_data/extracts/item_category.tsv"


def load_company_tuples(file: str = COMPANY_DATA):
    return pd.read_csv(file, sep="\t").values.tolist()


def load_item_tuples(file: str = ITEM_DATA):
    return pd.read_csv(file, sep="\t").values.tolist()


class KarmaDice:
    def __init__(self, company_pairs: list[tuple], item_pairs: list[tuple]):
        self.company_pairs = company_pairs
        self.item_pairs = item_pairs
        self.n_companies = len(company_pairs)
        self.n_items = len(item_pairs)

        self.company_counter = np.ones(self.n_companies)
        self.item_counter = np.ones(self.n_items)

    def get_random_company(self, size: int = 1):
        probs = 1 / self.company_counter
        probs = probs / probs.sum()
        ids = np.random.choice(range(self.n_companies), p=probs, size=size)
        return {k: self.company_pairs[k] for k in ids}

    def get_random_items(self, size: int = 4):
        probs = 1 / self.item_counter
        probs = probs / probs.sum()
        ids = np.random.choice(range(self.n_items), p=probs, size=size)
        return {k: self.item_pairs[k] for k in ids}

    def update_company_counter(self, ids: list[int]):
        self.company_counter[ids] += 1

    def update_item_counter(self, ids: list[int]):
        self.item_counter[ids] += 1


def create_prompt(
    company: str,
    company_category: str,
    brands: list[tuple]
) -> str:
    brand_list = [f"{idx + 1}. '{row[1]}' which sells '{row[0]}'" for idx, row in enumerate(brands)]
    brand_list = "\n".join(brand_list)

    content = (
        f"For the company '{company}' which is a '{company_category}' type company,\n"
        "Describe the company's previous purchase from each of these brands, there "
        "should be one sentence for each brand. The company name and or category "
        " should also be in each sentence.\n"
        f"{brand_list}"
    )
    return content


def load_openai_key():
    file = "/Users/timlee/Dropbox/keys/openai_key.txt"
    with open(file, "r") as f:
        return f.read()


def get_llm_responses(
    prompt_text: str,
):
    api_key = load_openai_key()
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a writing assistant, skilled in "
                    "drafting writing for wikipedia entries"
                )
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ]
    )
    return completion.choices[0].message.content


def get_hash_of_combination(company: str, brands: list[tuple]):
    """2nd column of brands is the name 1st is the category"""
    brandnames = [row[1] for row in brands]
    brandnames = "|".join(brandnames)
    str_phrase = f"{company}|{brandnames}"
    hashedname = md5(str_phrase.encode("utf-8")).hexdigest()
    return hashedname


def gen_llm_prompts_and_responses(n_prompts: int = 10, brands_per: int = 5):
    company_pairs = load_company_tuples()
    item_pairs = load_item_tuples()
    kd = KarmaDice(company_pairs, item_pairs)

    for j in range(n_prompts):
        company_lkup = kd.get_random_company()
        item_lkup = kd.get_random_items(size=brands_per)

        kd.update_company_counter(list(company_lkup.keys()))
        kd.update_item_counter(list(item_lkup.keys()))

        company_category, company = list(company_lkup.values())[0]
        brands = list(item_lkup.values())

        hashstr = get_hash_of_combination(company, brands)
        filename = "./sample_data/llm_generated/{}.json".format(hashstr)

        if os.path.exists(filename):
            print("combination already created, skipping: {}".format(filename))
            continue

        prompt_text = create_prompt(
            company=company,
            company_category=company_category,
            brands=brands
        )
        print("=" * 80)
        print(prompt_text)
        response_text = get_llm_responses(prompt_text)
        print(response_text)

        save_payload = dict(
            company=company,
            company_category=company_category,
            brands=brands,
            prompt_text=prompt_text,
            response_text=response_text,
        )

        with open(filename, "w") as f:
            f.write(json.dumps(save_payload, indent=2))
