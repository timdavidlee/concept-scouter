import locale
import numpy as np
import pandas as pd

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def topk_profile_probs(
    np_random: np.random.RandomState,
    k: int = 5
):
    flts = np_random.rand(k)
    flts = flts ** 2
    flts = flts / flts.sum()
    flts = np.sort(np.round(flts, 3))
    flts[-1] += 1 - flts.sum()
    return flts


def purchase_quantity_profile():
    qtys = np.array([1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100])
    probs = 1 / qtys
    probs = probs / probs.sum()
    return qtys, probs


def gen_country_profile():
    countries = {
        "United States": "25,462,700,000,000",
        "China": "17,963,200,000,000",
        "Japan": "4,231,140,000,000",
        "Germany": "4,072,190,000,000",
        "India": "3,385,090,000,000",
        "United Kingdom": "3,070,670,000,000",
        "France": "2,782,910,000,000",
        "Russia": "2,240,420,000,000",
        "Canada": "2,139,840,000,000",
        "Italy": "2,010,430,000,000",
        "Brazil": "1,920,100,000,000",
        "Australia": "1,675,420,000,000",
        "South Korea": "1,665,250,000,000",
        "Mexico": "1,414,190,000,000",
        "Spain": "1,397,510,000,000",
        "Indonesia": "1,319,100,000,000",
        "Saudi Arabia": "1,108,150,000,000",
        "Netherlands": "991,115,000,000",
        "Turkey": "905,988,000,000",
        "Switzerland": "807,706,000,000",
        "Poland": "688,177,000,000",
        "Argentina": "632,770,000,000",
        "Sweden": "585,939,000,000",
        "Norway": "579,267,000,000",
        "Belgium": "578,604,000,000",
        "Ireland": "529,245,000,000",
        "Israel": "522,033,000,000",
        "United Arab Emirates": "507,535,000,000",
        "Thailand": "495,341,000,000",
        "Nigeria": "477,386,000,000",
    }
    lkup = {c: locale.atoi(v) for c, v in countries.items()}
    p = np.array(list(lkup.values()))
    p = p / p.sum()
    countries = np.array(list(lkup.keys()))
    country_probs = p
    return countries, country_probs


def generate_date_profile(
    years: list[int]
):
    """Returns two arrays
    Returns:
        date_collector: a list of dates
        p: an array of probabilities totaling to 1
    """
    datecollector = []
    probs = []
    for y in years:
        start = "{}-01-01".format(y)
        end = "{}-12-31".format(y)

        datelist = pd.date_range(start, end, freq="d")
        x = np.linspace(0, 6.28, len(datelist))
        y = np.cos(x) + 2
        datecollector.extend(datelist)
        probs.extend(y)

    p = np.array(probs)
    p = p / p.sum()

    return datecollector, p


def random_trxn_ct_generator(
    intervals: int = 25,
    low: int = 3,
    high: int = 6
):
    # 1000 - 1M
    amounts = np.logspace(low, high, num=intervals)
    amounts = amounts.astype(int)
    probs = 1 / amounts
    probs = probs / probs.sum()
    return amounts, probs
