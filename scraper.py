from bs4 import BeautifulSoup
import requests
import pandas
import re
from datetime import datetime
from typing import List, Tuple


FACTORS = {"raw": ['bitcoin-marketcap', 'bitcoin-price', 'bitcoin-transactions', 'bitcoin-sentinusd', 'bitcoin-transactionvalue',
                   'bitcoin-mediantransactionvalue', 'bitcoin-transactionfees', 'bitcoin-median_transaction_fee',
                   'bitcoin-confirmationtime', 'bitcoin-size', 'bitcoin-fee_to_reward', 'bitcoin-difficulty',
                   'bitcoin-hashrate', 'bitcoin-mining_profitability', 'bitcoin-activeaddresses', 'google_trends-btc',
                   'top100cap-btc',
                   ],
           "technical": ['difficulty-btc', 'hashrate-btc', 'mining_profitability-btc',
                         'sentinusd-btc', 'transactionfees-btc', 'median_transaction_fee-btc',
                         'confirmationtime-btc', 'marketcap-btc', 'transactionvalue-btc',
                         'mediantransactionvalue-btc', 'fee_to_reward-btc',
                         'size-btc', 'transactions-btc', 'sentbyaddress-btc', 'top100cap-btc',
                         'activeaddresses-btc', 'price-btc'
                         ],
           }

PERIOD = ['3', '7', '14', '30', '90']
INDICATORS = ['sma', 'ema', 'wma', 'std', 'mom', 'var', 'trx', 'rsi', 'roc']
URL = "https://bitinfocharts.com/comparison/{}{}.html#alltime"
START_DATE = datetime(2013, 4, 1)
END_DATE = datetime(2017, 1, 1)
OUTPUT_PATH = "./data/uncleaned/"


def collect() -> None:
    col_names, vals, date = extract_data()
    df = pandas.DataFrame(list(zip(date, *vals)), columns=col_names)

    name = f'indicators-{START_DATE.strftime("%d-%m-%Y")}-{END_DATE.strftime("%d-%m-%Y")}'

    df.to_csv(f'{OUTPUT_PATH}/{name}.csv', index=False)


def extract_data() -> Tuple:
    # List (val) with values for every indicator is created. Then, this list is appended to the list of lists (vals)
    # which contains all indicators
    # TODO Date is being re-written after every scraped indicator. Replace?

    vals = []
    col_names = ["Date"]

    for type, factors in FACTORS.items():
        for factor in factors:

            if type == "technical":
                for indicator in INDICATORS:
                    for period in PERIOD:
                        tech_indicator_specs = f'-{indicator}{period}'
                        url = URL.format(factor, tech_indicator_specs)
                        col_name = f'{factor}-{tech_indicator_specs}'
                        val, date = scrape(url)

                        vals.append(val)
                        col_names.append(col_name)

            if type == "raw":
                url = URL.format(factor, '')
                col_name = f'{factor}_raw'
                val, date = scrape(url)

                vals.append(val)
                col_names.append(col_name)

    return col_names, vals, date


def scrape(url: str) -> Tuple:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    scripts = soup.find_all('script')
    for script in scripts:
        if 'd = new Dygraph(document.getElementById("container")' in script.text:
            content = script.text
            content = '[[' + content.split('[[')[-1]
            content = content.split(']]')[0] + ']]'
            content = content.replace("new Date(", '').replace(')', '')
            data = parse_content(content)

    date = []
    val = []
    day = None
    for obs in data:
        if (data.index(obs) % 2) == 0:
            day = datetime.strptime(obs, "%Y/%m/%d")
            if day < START_DATE:
                continue
            elif day > END_DATE:
                break
            date.append(obs)
        else:
            if day < START_DATE:
                continue
            elif day > END_DATE:
                break
            val.append(obs)

    return val, date


def parse_content(content: str) -> List:
    clean = re.sub("[\[\],\s]", "", content)  # noqa
    split = re.split("[\'\"]", clean)
    parsed = [s for s in split if s != '']
    return parsed


collect()
