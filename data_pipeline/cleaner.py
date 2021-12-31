import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(">> Cleaning dataset")
    df = df.interpolate(limit_direction="both", limit=20)
    df = df.fillna(df.mode().iloc[0])

    print(">>> Done cleaning")

    return df
