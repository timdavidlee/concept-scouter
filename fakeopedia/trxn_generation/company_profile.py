import numpy as np
from scipy.stats import pareto

from fakeopedia.trxn_generation.stats import generate_date_profile


class CompanyProfile:
    def __init__(
        self,
        company_id: int,
        categories: list[str],
        cat_probs: list[float],
        countries: list[str],
        country_probs: list[float],
        cat2iid: dict[str, list],
        years: list[int] = None,
    ):
        if not years:
            # generate transactions for 3 years
            self.years = [2021, 2022, 2023]
        else:
            self.years = years

        self.cid = company_id
        # will be a nested dictionary
        self.profiles = dict()

        self.cat_probs = np.array(cat_probs)
        self.categories = np.array(categories)

        self.country_probs = np.array(country_probs)
        self.countries = countries

        self.cat2iid = cat2iid

    @staticmethod
    def generate_pareto_quantities(
        target_n: int = 100_000,
        n_items: int = 1000
    ):
        """Generates a descending distribution of items purchased
        This was designed so that each company would purchase
        a dominant item out of a subcategory and have descending
        popularity.
        """
        x = np.linspace(1, 100, n_items)
        vals = pareto.pdf(x, b=1)

        factor = target_n / vals.sum()
        z = (vals * factor).astype(int) + 1
        return z

    def calculate_trxn(
        self,
        np_random: np.random.RandomState,
        total_qty: int = 100_000
    ) -> list:
        qty_per_cat = (self.cat_probs * total_qty).astype(int)
        effective_total = 0

        datelist, date_probs = generate_date_profile(self.years)

        transactions = []
        for c, qc in zip(self.categories, qty_per_cat):
            iids = self.cat2iid[c]
            inds = np.arange(len(iids))
            np_random.shuffle(inds)

            purchased_qtys = self.generate_pareto_quantities(qc)
            # very difficult to get exact
            cat_effective_total = purchased_qtys.sum()
            effective_total += cat_effective_total
            n_qtys = len(purchased_qtys)
            purchased_iids = inds[:n_qtys]

            self.profiles[c] = dict(iids=purchased_iids, qtys=purchased_qtys)

            # expand item, qty pairs into a full list, then generate the other 
            # attributes
            items_purchased = self.expand(
                np_random,
                purchased_iids,
                purchased_qtys
            )

            # repeat enough values for company
            company_id_arr = np.repeat(self.cid, cat_effective_total)

            # repeat each values for category
            category_arr = np.repeat(c, cat_effective_total)

            countries = np_random.choice(
                self.countries,
                p=self.country_probs,
                size=cat_effective_total
            )

            date_arr = np_random.choice(
                datelist,
                p=date_probs,
                size=cat_effective_total
            )

            transactions += list(
                list(zip(company_id_arr, category_arr, items_purchased, countries, date_arr))
            )

        return transactions

    @staticmethod
    def expand(
        np_random: np.random.RandomState,
        labels: list,
        cts: list[int]
    ):
        """
        Takes in:
            ["A", "B", "C"], [5, 2, 1]

        Returns:
            ['C', 'A', 'A', 'A', 'B', 'A', 'B', 'A']
        """
        collector = []
        for lbl, c in zip(labels, cts):
            collector += [lbl] * c

        np_random.shuffle(collector)
        return collector
