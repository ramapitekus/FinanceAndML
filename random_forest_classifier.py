import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main():

    df = pd.read_csv("./data/cleaned/cleaned_indicators-01-04-2013-01-01-2017.csv")
    print(df.shape)
    x = df.drop(labels=['bitcoin-price_raw', 'Date'], axis=1)
    # TODO edit the dataset such that y is 1d/7d/30d/90d ahead from features
    # Keep only important features, then add Variance threshold from the notebook
    # In the end, we will have 4 separate datasets (for every time horizon we want to predict) with features
    y = df['bitcoin-price_raw']
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(x, y)

    # TODO add random forest classifier for categorical dependent variable (i.e., 1 for price up, 0 down)
    return None


main()
