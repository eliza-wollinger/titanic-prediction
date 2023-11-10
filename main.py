import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("/content/titanic_data.csv")
data.head()

data = data.drop(["zero", "zero.1", "zero.2", "zero.3", "zero.4", "zero.5", "zero.6", "zero.7", "zero.8",
                  "zero.9", "zero.10", "zero.11", "zero.12", "zero.13", "zero.14", "zero.15", "zero.16",
                  "zero.17", "zero.18", "Embarked"], axis = 1)
data.head()

data = data.set_index(['Passengerid'])
data = data.rename(columns = {'2urvived' : 'target'}, inplace = False)
data.head()

data.describe()
data.isnull().sum()

x_train, x_test, y_train, y_test = train_test_split(data.drop(['target'], axis = 1),
                                                    data['target'],
                                                    test_size = 0.3,
                                                    random_state = 1234)

[{'train' : x_train.shape}, {'test' : x_test.shape}]

# Mapping the model
rndforest = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', max_depth = 5)

# Calculating the model
rndforest.fit(x_train, y_train)

probability    = rndforest.predict_proba(data.drop('target', axis = 1))[:,1]
classification = rndforest.predict(data.drop('target', axis = 1))
data
