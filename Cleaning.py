import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/paulo/Desktop/Master 2 /Machine Learning/data/indicators-01-04-2013-31-12-2020.csv')

def bit_of_cleaning(df):
    
    df.rename(columns={'Unnamed: 0': 'Index'},inplace=True)
    df.set_index('Index', inplace=True)
   
    return df 

df = bit_of_cleaning(df)

# Distribution of N/A before dropping columns : 
    
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis', cbar=False)
plt.savefig('/Users/paulo/Desktop/na_before.pdf')

# Remove the whole columns when N/A are too important (More than 5% by variable)

def remove_na(df) :
    
    df_1 = df.loc[:, (df.isnull().mean(axis=0) <= 0.05)] # We can play with this threeshold
    df_2 = df.loc[:, (df.isnull().mean(axis=0) > 0.05)]
    
    print('The removed columns are :' )
    for col in df_2.columns:
        print(col)
         
    return df_1

df = remove_na(df)

# Distribution of N/A after dropping columns :
    
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis', cbar=False)
plt.savefig('/Users/paulo/Desktop/na_after.pdf')


# Fill N/A with linear interpolation 

def fill_na_linear(df):
    df = df.interpolate(method='linear', limit_direction ='forward')
    return df


# Fill N/A with last available value 

def fill_na_last(df):
    df = df.interpolate(method='pad')
    return df


