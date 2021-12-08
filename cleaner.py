import pandas as pd
import os


PATH = "./data/uncleaned/"
OUTPUT_PATH = "./data/cleaned/"


def main():

    for csv in os.listdir(PATH):
        df = pd.read_csv(f"{PATH}/{csv}")
        df = df.interpolate(limit_direction="both", limit=20)
        df = df.fillna(df.mode().iloc[0])
        df.to_csv(f"{OUTPUT_PATH}/cleaned_{csv}", index=False)


main()
