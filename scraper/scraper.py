from bs4 import BeautifulSoup
import requests
import pandas
import re
import os
from typing import Tuple, List


FACTORS = ['bitcoin-marketcap', 'bitcoin-price', 'transactions', 'bitcoin-sentinusd', 'bitcoin-transactionvalue',
           'bitcoin-mediantransactionvalue', 'bitcoin-transactionfees', 'bitcoin-median_transaction_fee',
           'bitcoin-confirmationtime', 'bitcoin-size', 'bitcoin-fee_to_reward', 'bitcoin-difficulty',
           'bitcoin-hashrate', 'bitcoin-mining_profitability', 'bitcoin-activeaddresses', 'bitcoin-tweets']

URL = "https://bitinfocharts.com/comparison/{}.html#alltime"


def main() -> None:
    for factor in FACTORS:
        date, val = scrape(url=URL.format(factor))
        write_to_csv(date, val, factor)


def scrape(url: str) -> Tuple:
    flag = 0
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    scripts = soup.find_all('script')
    for script in scripts:
        if 'd = new Dygraph(document.getElementById("container")' in script.text:
            flag = 1
            content = script.text
            content = '[[' + content.split('[[')[-1]
            content = content.split(']]')[0] + ']]'
            content = content.replace("new Date(", '').replace(')', '')
            data = parse_content(content)

    if not flag:
        raise AssertionError("Factor not found")

    date = []
    val = []
    for obs in data:  # noqa
        if (data.index(obs) % 2) == 0:
            date.append(obs)
        else:
            val.append(obs)
    return date, val


def parse_content(content: str) -> List:
    clean = re.sub("[\[\],\s]", "", content)
    split = re.split("[\'\"]", clean)
    parsed = [s for s in split if s != '']
    return parsed


def write_to_csv(date: List, val: List, factor: str) -> None:
    df = pandas.DataFrame(list(zip(date, val)), columns=["Date", "val"])
    if not os.path.isdir('./scraper/data'):
        os.mkdir('./scraper/data')

    df.to_csv(f'./scraper/data/{factor}.csv')  # noqa


main()
