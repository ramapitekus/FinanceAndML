import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


PERIODS = [1, 7, 30, 90]

TIME_PERIODS = ["01-04-2013-01-01-2017", "01-04-2013-19-07-2016", "01-04-2013-31-12-2020"]


def main() -> None:

    for date in TIME_PERIODS:
        df = pd.read_csv(f"./data/binary_price_change_cols/indicators-{date}.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        for period in PERIODS:

            col_format = "{}d_binary_price_change"
            y_col_name = col_format.format(period)
            shifted_values = [col_format.format(p) for p in PERIODS]
            x = df.drop(columns=shifted_values)
            y = df[y_col_name]

            x_important_cols = feature_selection(x, y, period)

            df_filtered = df[list(x_important_cols) + shifted_values]

            df_filtered.to_csv(f"./data/binary_price_change_cols/filtered/{date}/{period}d_indicators.csv")

    return None


def feature_selection(x: pd.DataFrame, y: pd.DataFrame, period: int) -> pd.DataFrame:
    print(f"period {period}")
    x, y = x.iloc[:-period, :], y[:-period]  # Do not fit on the NAs because of shift
    classifier = SelectFromModel(RandomForestClassifier(n_estimators=300))
    classifier.fit(x, y)

    x_important_cols = x.columns[classifier.get_support()]

    return x_important_cols


main()
