# analysis.py
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV from data folder 
df = pd.read_csv("data/powerball_Post_Oct_2015.csv")
df.columns = df.columns.str.strip()

# File parse Tests:
#print(df.head())
#print(df.columns)

# Column header consistentcy with loop check. 
#print(df.columns.tolist())

# Number Count Test
#for col in ['Ball 1','Ball 2','Ball 3','Ball 4','Ball 5','Powerball']:
#    print(df[col].value_counts().sort_index())


# Powerball Plot 
df['Powerball'].value_counts().sort_index().plot(kind='bar', figsize=(12,6), title='Powerball Number Frequencies')
plt.xlabel('Powerball Number')
plt.ylabel('Occurrences')
plt.show()

df['Ball 1'].value_counts().sort_index().plot(kind="bar", figsize=(12,6), title="Ball 1 Number Frequencies")
plt.xlabel('Ball 1 Number')
plt.ylabel('Occurences')
plt.show()