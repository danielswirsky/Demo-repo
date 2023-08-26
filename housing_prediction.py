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

#Feature Engineering

from sklearn.preprocessing import StandardScaler

#scale the data 
scaler = StandardScaler()

X = df.drop('price', axis='columns')
y = df.price

X_scaled = scaler.fit_transform(X)

#split data in a train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# regr = LinearRegression()
# regr.fit(X_train, y_train)
# print(regr.score(X_test,y_test))

from sklearn.neighbors import KNeighborsRegressor
# knn = KNeighborsRegressor()
# knn.fit(X_train, y_train)
# print(knn.score(X_test, y_test))

from sklearn.tree import DecisionTreeRegressor
# tree = DecisionTreeRegressor()
# tree.fit(X_train, y_train)
# print(tree.score(X_test, y_test))

from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()
# rf.fit(X_train, y_train)
# print(rf.score(X_test,y_test))

model_params = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'ridge_regression': {
        'model': Ridge(),
        'params': {
            'alpha': [1.0, 10, 20],
            'solver': ['auto', 'svd', 'cholesky']
        }
    },
    'knn_regression': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [5, 10, 20],
            'weights': ['uniform', 'distance']
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'criterion': ['squared_error', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random']
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [100, 250, 400],
            'criterion': ['squared_error', 'absolute_error', 'poisson']
        }
    }
}

from sklearn.model_selection import GridSearchCV
scores = []

for name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
clf_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(clf_df)