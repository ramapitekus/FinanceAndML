import json
from scraper import scrape
from cleaner import clean
from feature_selection import feature_selection

SETTINGS = open("./data_pipeline/settings.json")
SETTINGS_DICT = json.load(SETTINGS)


def main() -> None:

    scrape(SETTINGS_DICT)
    clean()

    # Perform feature selection for categorical dependent variable
    feature_selection(SETTINGS_DICT,
                      categorical=True,
                      vif_threshold=15,
                      rf_threshold=1e-4)

    # Perform feature selection for continuous dependent variable
    feature_selection(SETTINGS_DICT,
                      categorical=True,
                      vif_threshold=15,
                      rf_threshold=1e-4)


if __name__ == "__main__":
    main()
