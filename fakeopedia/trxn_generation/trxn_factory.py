from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger
from humanfriendly import format_timespan

from fakeopedia.trxn_generation.company_profile import CompanyProfile
from fakeopedia.trxn_generation.stats import (
    purchase_quantity_profile,
    gen_country_profile,
    topk_profile_probs,
    random_trxn_ct_generator
)


class TransactionFactory:
    def __init__(
        self,
        cat2iid: dict[str, list],
        cats_per: int = 5,
        countries_per: int = 4,
        n_transactions: int = None,
        purchase_batch_sz: int = 5,
        sample_company_size: int = None,
        n_total_companies: int = 73_974,
        n_jobs: int = 8,
    ):
        self.n_jobs = n_jobs
        self.np_random = np.random.RandomState()
        self.cats_per = cats_per
        self.countries_per = countries_per

        if n_transactions is None:
            self.trxn_cts, self.trxn_probs = random_trxn_ct_generator()
        else:
            # if provided, will be constant
            if not isinstance(n_transactions, int):
                raise ValueError("not an int: {}".format(n_transactions))
            self.trxn_cts = [n_transactions]
            self.trxn_probs = np.array([1.0])

        self.purchase_batch_sz = purchase_batch_sz

        self.np_random = np.random.RandomState()
        self.cat2iid = cat2iid
        self.item_categories = sorted(list(cat2iid.keys()))
        self.sample_company_size = sample_company_size or n_total_companies
        self.n_total_companies = n_total_companies
        self.qty, self.qty_probs = purchase_quantity_profile()

        self.company_profiles = dict()
        self._init_company_profiles()
        self.transactions = []

    def _init_company_profiles(self):
        countries, country_probs = gen_country_profile()
        n_countries = len(countries)

        for k in range(self.sample_company_size):
            company_id = self.np_random.randint(self.n_total_companies)
            logger.info(f"initing company: {company_id}")

            top_cats = self.np_random.choice(self.item_categories, size=self.cats_per)
            top_probs = topk_profile_probs(self.np_random, k=self.cats_per)
            inds = self.np_random.choice(range(n_countries), p=country_probs, size=self.countries_per)

            top_countries = countries[inds]
            top_country_probs = country_probs[inds] / country_probs[inds].sum()

            comp_profile = CompanyProfile(
                company_id,
                top_cats,
                top_probs,
                top_countries,
                top_country_probs,
                self.cat2iid
            )
            self.company_profiles[company_id] = comp_profile

    def generate_transactions(self):
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = []
            for cid in self.company_profiles:
                futures.append(pool.submit(self._generate_trxn_for_company, cid=cid))

            for future in as_completed(futures):
                self.transactions.extend(future.result())

        total_transactions = len(self.transactions)
        logger.info("records {:,} total runtime: {}".format(
            total_transactions,
            format_timespan(perf_counter() - start))
        )

    def _generate_trxn_for_company(self, cid: int):
        n_transactions = self.np_random.choice(self.trxn_cts, p=self.trxn_probs)
        start_ts = perf_counter()
        logger.info(f"generating transactions for company: {cid}")
        comp_profile = self.company_profiles[cid]
        local_trxns = comp_profile.calculate_trxn(self.np_random, n_transactions)
        logger.info("{} -> runtime: {}".format(cid, format_timespan(perf_counter() - start_ts)))
        return local_trxns
