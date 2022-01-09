from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import perform_statistics_classification


INTERVALS = {
    1: "01-04-2013-19-07-2016",
    2: "01-04-2013-01-01-2017",
    3: "01-04-2013-31-12-2020",
}
PERIODS = [1, 7, 30, 90]


def train_interval(interval: int) -> None:
    for period in PERIODS:
        interval_str = INTERVALS[interval]

        standard_scaler = StandardScaler()
        df = pd.read_csv(
            f"./data/categorical/{interval_str}/{period}d_indicators.csv",
            index_col=False,
        )

        y = df[f"{period}d_binary_price_change"]

        x = df.drop(columns=[f"{period}d_binary_price_change", "Date"])  # Get features
        x = x.values[:-period]

        y = df[f"{period}d_binary_price_change"]
        y = y.values[:-period]  # Get dependent variables

        x = standard_scaler.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False, random_state=3
        )

        print(f"Training interval {interval} for {period} day(s) period")

        svr = SVC(C=8000)
        fitted = svr.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)

        perform_statistics_classification(
            y_test, y_pred, model_type="svm", interval=interval, period=period
        )


for interval in INTERVALS.keys():
    train_interval(interval)
