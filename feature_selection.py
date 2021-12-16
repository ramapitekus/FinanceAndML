import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from functools import wraps
from time import time
from statsmodels.stats.outliers_influence import variance_inflation_factor

# TODO perhaps add multiprocessing

PERIODS = [1, 7, 30, 90]
DATES = ["01-04-2013-01-01-2017", "01-04-2013-19-07-2016", "01-04-2013-31-12-2020"]
INPUT_PATH = {True: "./data/all_features/categorical", False: "./data/all_features/continuous"}
OUTPUT_PATH = {True: "./data/features_selected/categorical/{}/{}d_indicators.csv", False: "./data/features_selected/continuous/{}/{}d_indicators.csv"}


def feature_selection(categorical: bool, vif_threshold: float, rf_threshold: float) -> None:

    for date in DATES:
        for period in PERIODS:

            df = pd.read_csv(f"{INPUT_PATH[categorical]}/indicators-{date}.csv")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

            col_format = "{}d_binary_price_change" if categorical else "bitcoin-price_raw_{}d"

            df_filtered = random_forest(df,
                                        col_format,
                                        rf_threshold,
                                        date=date,
                                        period=period,
                                        categorical=categorical)

            df_filtered = VIF_feature_selection(df_filtered,
                                                col_format,
                                                vif_threshold,
                                                date=date,
                                                period=period,
                                                categorical=categorical
                                                )

            df_filtered.to_csv(OUTPUT_PATH[categorical].format(date, period))


def timer(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start = time()
        dependent_type = "categorical" if kwargs['categorical'] else "continuous"
        print(f"> Running {orig_func.__name__} feature selection for {kwargs['period']} day(s) prediction on {dependent_type} dependent variable on period {kwargs['date']}.")
        df = orig_func(*args, **kwargs)
        end = time()
        print(f">> {orig_func.__name__} took {end-start} seconds.")
        return df

    return wrapper


@timer
def random_forest(df: pd.DataFrame, col_format: str, threshold: float, date: str, period: int, categorical: bool) -> pd.DataFrame:
    y_col_name = col_format.format(period)
    shifted_values = [col_format.format(p) for p in PERIODS]
    x = df.drop(columns=shifted_values)
    y = df[y_col_name]

    x, y = x.iloc[:-period, :], y[:-period]  # Do not fit on the NAs because of shift
    if categorical:
        model = SelectFromModel(RandomForestClassifier(n_estimators=100))
        model.fit(x, y)

    else:
        model = SelectFromModel(RandomForestRegressor(n_estimators=100), threshold=threshold)
        model.fit(x, y)

    x_important_cols = list(x.columns[model.get_support()])

    return df[x_important_cols + shifted_values]


@timer
def VIF_feature_selection(df: pd.DataFrame, col_format: str, threshold: float, date: str, period: int, categorical: bool) -> pd.DataFrame:
    dependent_columns = [col_format.format(period) for period in PERIODS]
    x = df.drop(columns=dependent_columns)

    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

    relevant_features = []
    for _, row in vif_data.iterrows():
        if row["VIF"] < threshold:
            relevant_features.append(row["feature"])

    return df[relevant_features + dependent_columns]


feature_selection(categorical=False, vif_threshold=15, rf_threshold=1e-4)
feature_selection(categorical=True, vif_threshold=15, rf_threshold=1e-4)
