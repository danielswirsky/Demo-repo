import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\danie\OneDrive\Desktop\Programming\Learning ML\Housing Portfolio Proj\Housing.csv")
#print(df.head())

#Check for Null values
#print(df.info())

print(df[df['area'] < (df.area.mean() + (-3*df.area.std()))])
print(df[df.area > (df.area.mean() + (3*df.area.std()))])