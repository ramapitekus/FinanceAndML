import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


PERIODS = [1, 7, 30, 90]

TIME_PERIODS = ["01-04-2013-01-01-2017", "01-04-2013-19-07-2016", "01-04-2013-31-12-2020"]


def main() -> None:

    for date in TIME_PERIODS:
        df = pd.read_csv(f"./data/binary_price_change_cols/indicators-{date}.csv")

        for period in PERIODS:

            col_format = "{}d_binary_price_change"
            y_col_name = col_format.format(period)
            col_to_drop = [col_format.format(p) for p in PERIODS]
            x = df.drop(columns=col_to_drop + ["Date"])
            y = df[y_col_name]

            x_important_cols = feature_selection(x, y, period)

            df_filtered = df[x_important_cols]

            df_filtered.to_csv(f"./data/binary_price_change_cols/filtered/{date}/{period}d_indicators.csv", index=False)

    return None


def feature_selection(x: pd.DataFrame, y: pd.DataFrame, period: int) -> pd.DataFrame:
    print(f"period {period}")
    x = x.iloc[:, :-period]
    classifier = SelectFromModel(RandomForestClassifier(n_estimators=300))
    classifier.fit(x, y)

    x_important_cols = x.columns[classifier.get_support()]

    return x_important_cols


main()
