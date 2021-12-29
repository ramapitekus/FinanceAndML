from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from utils import perform_statistics
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


INTERVALS = {1: "01-04-2013-19-07-2016", 2: "01-04-2013-01-01-2017", 3: "01-04-2013-31-12-2020"}
PERIODS = [1, 7, 30, 90]


def train_interval(interval: int) -> None:
    for period in PERIODS:
        interval_str = INTERVALS[interval]

        # min_max_scaler = MinMaxScaler()
        # robust_scaler = RobustScaler()
        standard_scaler = StandardScaler()
        df = pd.read_csv(f"./data/features_selected/continuous/{interval_str}/{period}d_indicators.csv", index_col=False)

        y = df[f"bitcoin-price_raw_{period}d"]
        
        x = df.drop(columns=[f"bitcoin-price_raw_{period}d", "Date"])  # Get features
        x = x.values[:-period]

        y = df[f"bitcoin-price_raw_{period}d"]
        y = y.values[:-period]  # Get dependent variables

        # x = min_max_scaler.fit_transform(x)
        x = standard_scaler.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        print(f"Training interval {interval} for {period} day(s) period")

        svr = SVR(kernel='rbf',C=10000)
        fitted = svr.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)

        # mse = mean_squared_error(y_pred, y_test)
# 
        # print(f"MSE for period {period} is {mse}")

        perform_statistics(y_pred, y_test)


for interval in INTERVALS.keys():
    train_interval(interval)