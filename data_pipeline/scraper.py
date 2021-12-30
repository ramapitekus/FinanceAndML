from bs4 import BeautifulSoup
import requests
import pandas
import re
from datetime import datetime
from typing import List, Tuple


URL = "https://bitinfocharts.com/comparison/{}{}.html#alltime"
OUTPUT_PATH = "./data/uncleaned/"


def scrape(settings: dict) -> None:

    for start_date_str, end_date_str in settings["dates"]:
        print(f">> Collecting data for period {start_date_str} to {end_date_str}")
        start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
        end_date = datetime.strptime(end_date_str, "%d-%m-%Y")

        col_names, vals, date = extract_data(settings["factors"],
                                             settings["indicators"],
                                             settings["periods"],
                                             start_date,
                                             end_date,
                                             )

        df = pandas.DataFrame(list(zip(date, *vals)), columns=col_names)
        name = f'indicators-{start_date_str}-{end_date_str}'
        df.to_csv(f'{OUTPUT_PATH}/{name}.csv', index=False)

    print(">>> Done scraping")

    return None


def extract_data(factors: dict,
                 indicators: List,
                 periods: List,
                 start_date: datetime,
                 end_date: datetime
                 ) -> Tuple:
    # List (val) with values for every indicator is created. Then, this list is appended to the list of lists (vals)
    # which contains all indicators
    # TODO Date is being re-written after every scraped indicator. Replace?
    # TODO Would be faster to scrape the longest period and split the datasets according to sub-periods

    vals = []
    col_names = ["Date"]

    for type, factors in factors.items():
        for factor in factors:

            if type == "technical":
                for indicator in indicators:
                    for period in periods:
                        tech_indicator_specs = f'-{indicator}{period}'
                        url = URL.format(factor, tech_indicator_specs)
                        col_name = f'{factor}-{tech_indicator_specs}'
                        val, date = collect(url, start_date, end_date)

                        vals.append(val)
                        col_names.append(col_name)

            if type == "raw":
                url = URL.format(factor, '')
                col_name = f'{factor}_raw'
                val, date = collect(url, start_date, end_date)

                vals.append(val)
                col_names.append(col_name)

    return col_names, vals, date


def collect(url: str, start_date: datetime, end_date: datetime) -> Tuple:
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
            if day < start_date:
                continue
            elif day > end_date:
                break
            date.append(obs)
        else:
            if day < start_date:
                continue
            elif day > end_date:
                break
            val.append(obs)

    return val, date


def parse_content(content: str) -> List:
    clean = re.sub("[\[\],\s]", "", content)  # noqa
    split = re.split("[\'\"]", clean)
    parsed = [s for s in split if s != '']
    return parsed
