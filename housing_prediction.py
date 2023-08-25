import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\danie\OneDrive\Desktop\Programming\Learning ML\Housing Portfolio Proj\Housing.csv")
#print(df.head())

#EXPLORATORY DATA ANALYSIS

#Check for Null values
#print(df.info())

#print(df.shape)
#print(df[df['area'] < (df.area.mean() + (-3*df.area.std()))])
#print(df[df.area > (df.area.mean() + (3*df.area.std()))])
df = df[df.area < (df.area.mean() + (3*df.area.std()))]
#print(df.shape)

#print(df.dtypes)
#print(df.furnishingstatus.unique())

#Can manually replace object colums
#or use pandas get_dummies for OHE
df.mainroad.replace(
    {'yes': 1, 'no': 0}, inplace=True
)
df.guestroom.replace(
    {'yes': 1, 'no': 0},inplace= True 
)
df.basement.replace(
    {'yes': 1, 'no': 0}, inplace=True
)
df.hotwaterheating.replace(
    {'yes': 1, 'no': 0}, inplace=True
)
df.airconditioning.replace(
    {'yes': 1, 'no': 0}, inplace=True 
)
df.prefarea.replace(
    {'yes': 1, 'no': 0}, inplace=True
)

df = pd.get_dummies(df, drop_first=True)
#print(df.dtypes)
#print(df.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = df.drop('price', axis='columns')
y = df.price

X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=10)

print(len(X_train))
print(len(X_test))