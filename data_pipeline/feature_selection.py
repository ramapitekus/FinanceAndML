import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from functools import wraps
from time import time
from statsmodels.stats.outliers_influence import variance_inflation_factor


def feature_selection(
    df: pd.DataFrame,
    settings: dict,
    period: int,
    date: str,
    categorical: bool,
    vif_threshold: float,
    rf_threshold: float,
) -> pd.DataFrame:

    col_name = "{}d_binary_price_change" if categorical else "bitcoin-price_raw_{}d"

    df_filtered = random_forest(
        df,
        col_name,
        rf_threshold,
        settings,
        date=date,
        period=period,
        categorical=categorical,
    )

    df_filtered = VIF_feature_selection(
        df_filtered,
        col_name,
        vif_threshold,
        date=date,
        period=period,
        categorical=categorical,
    )

    return df_filtered


def timer(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start = time()
        dependent_type = "categorical" if kwargs["categorical"] else "continuous"
        print(
            f"> Running {orig_func.__name__} feature selection for {kwargs['period']} day(s) prediction on {dependent_type} dependent variable on period {kwargs['date']}."
        )
        df = orig_func(*args, **kwargs)
        end = time()
        print(f">> {orig_func.__name__} took {end-start} seconds.")
        return df

    return wrapper


@timer
def random_forest(
    df: pd.DataFrame,
    col_name: str,
    threshold: float,
    settings: dict,
    date: str,
    period: int,
    categorical: bool,
) -> pd.DataFrame:

    # Do not fit on last x days, because of shift (shift causes last x values to be NAs)
    df = df.iloc[:-period, :]
    # Set dependent y col name for given period
    y_col_name = col_name.format(period)
    # Remove all remaining dependent cols (y) of different periods from the x dataset
    dependent_cols = [col_name.format(p) for p in settings["pred_periods"]]
    x = df.drop(columns=dependent_cols)
    y = df[y_col_name]

    if categorical:
        model = SelectFromModel(
            RandomForestClassifier(n_estimators=100), threshold=threshold
        )
        model.fit(x, y)

    else:
        model = SelectFromModel(
            RandomForestRegressor(n_estimators=100), threshold=threshold
        )
        model.fit(x, y)

    relevant_features = list(x.columns[model.get_support()])

    return df[relevant_features + [y_col_name]]


@timer
def VIF_feature_selection(
    df: pd.DataFrame,
    col_name: str,
    threshold: float,
    date: str,
    period: int,
    categorical: bool,
) -> pd.DataFrame:

    y_col_name = col_name.format(period)
    x = df.drop(columns=y_col_name)

    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["VIF"] = [
        variance_inflation_factor(x.values, i) for i in range(len(x.columns))
    ]

    relevant_features = []
    for _, row in vif_data.iterrows():
        if row["VIF"] < threshold:
            relevant_features.append(row["feature"])

    return df[relevant_features + [y_col_name]]
