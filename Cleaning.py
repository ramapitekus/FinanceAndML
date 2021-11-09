import pandas as pd
import seaborn as sns
import os
from os import path
import matplotlib.pyplot as plt


FILE_NAME = "indicators-01-04-2013-31-12-2020.csv"
PATH = "./scraper/data/{}"
THRESHOLD = 0.05


def main():
    if not path.exists("./cleaned"):
        os.mkdir("./cleaned")
    # path = PATH.format(FILE_NAME)
    df = pd.read_csv(path)
    cleaned_df = bit_of_cleaning(df)
    # Distribution of N/A before dropping columns
    make_visualizations(f"before_drop_{FILE_NAME[:-4]}", cleaned_df)
    df_removed_na = remove_na(f"{FILE_NAME}", df)
    # Distribution of N/A after dropping columns :
    make_visualizations(f"after_drop_{FILE_NAME[:-4]}", df_removed_na)

    save_to_csv(df_removed_na)


def bit_of_cleaning(df):
    df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    df.set_index('Index', inplace=True)
   
    return df


def remove_na(name, df):
    # Remove the whole columns when N/A are too important (More than 5% by variable)
    df_1 = df.loc[:, (df.isnull().mean(axis=0) <= THRESHOLD)]
    df_2 = df.loc[:, (df.isnull().mean(axis=0) > THRESHOLD)]
    
    print(f'The removed columns for {name} are :')
    for col in df_2.columns:
        print(col)
         
    return df_1


def fill_na_linear(df):
    # Fill N/A with linear interpolation
    df = df.interpolate(method='linear', limit_direction ='forward')
    return df


def fill_na_last(df):
    # Fill N/A with last available value
    df = df.interpolate(method='pad')
    return df


def make_visualizations(name: str, df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis', cbar=False)
    plt.savefig(f'./{name}.pdf')


def save_to_csv(df):
    if not path.exists("./cleaned"):
        os.mkdir("./cleaned")
    df.to_csv(f"./cleaned/cleaned_{FILE_NAME}")


main()
