import pandas as pd
from typing import List


def perform_shift(df: pd.DataFrame, periods: List) -> pd.DataFrame:
    for period in periods:
        df[f"bitcoin-price_raw_{period}d"] = df.iloc[:, 1].shift(-period)

    return df


def create_target_columns(df: pd.DataFrame, periods: List) -> pd.DataFrame:
    df_shifted = perform_shift(df, periods)

    # Shifted df can be used directly for continuous feature selection
    # Use method .copy() such that both cont. and cat. DF do not point to the same dataframe
    # Which would cause for both variables to have both categorical and continuous dependent variables

    df_continuous = df_shifted.copy()
    df_categorical = df_shifted.copy()

    # Add categorical columns (price rise, fall) to categorical df
    # Finally, drop shifted bitcoin prices, since these are not necessary
    for period in periods:
        df_categorical[f"bitcoin-price_raw_{period}d"] = df_categorical.iloc[:, 1].shift(-period)

        df_categorical.loc[
            df_categorical[f"bitcoin-price_raw_{period}d"] >= df_categorical["bitcoin-price_raw"],
            f"{period}d_binary_price_change",
        ] = 1
        df_categorical.loc[
            df_categorical[f"bitcoin-price_raw_{period}d"] < df_categorical["bitcoin-price_raw"],
            f"{period}d_binary_price_change",
        ] = 0

        df_categorical.drop(columns=[f"bitcoin-price_raw_{period}d"], inplace=True)

    return df_continuous, df_categorical
