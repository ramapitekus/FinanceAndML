import json
from scraper import scrape
from cleaner import clean
from feature_selection import feature_selection
from time_shifting import create_target_columns


SETTINGS = json.load(open("./data_pipeline/settings.json"))

OUTPUT_PATH = {
    "categorical": "./data/categorical/{}/{}d_indicators.csv",
    "continuous": "./data/continuous/{}/{}d_indicators.csv",
}


def main() -> None:

    for start_date_str, end_date_str in SETTINGS["dates"]:
        date_str = start_date_str + "-" + end_date_str

        scraped_df = scrape(SETTINGS, start_date_str, end_date_str)
        cleaned_df = clean(scraped_df)

        df_continuous, df_categorical = create_target_columns(cleaned_df, SETTINGS["pred_periods"])

        for pred_period in SETTINGS["pred_periods"]:

            # Perform feature selection for continuous dependent variable
            features_selected_cont_df = feature_selection(
                df_continuous,
                SETTINGS,
                pred_period,
                date_str,
                categorical=False,
                vif_threshold=SETTINGS["vif_threshold_continuous"],
                rf_threshold=SETTINGS["rf_threshold_continuous"],
            )

            # Perform feature selection for categorical dependent variable
            features_selected_cat_df = feature_selection(
                df_categorical,
                SETTINGS,
                pred_period,
                date_str,
                categorical=True,
                vif_threshold=SETTINGS["vif_threshold_categorical"],
                rf_threshold=SETTINGS["rf_threshold_categorical"],
            )

            features_selected_cont_df.to_csv(
                OUTPUT_PATH["continuous"].format(date_str, pred_period), index=True
            )
            features_selected_cat_df.to_csv(
                OUTPUT_PATH["categorical"].format(date_str, pred_period), index=True
            )


if __name__ == "__main__":
    main()
