# Based on the tutorial from here: https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

import pandas as pd
import numpy as np
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

def preprocess(df):
    # Map female -> 0 and male -> 1
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Map embarkation ports to features
    df['Cport'] = df['Embarked'].map(lambda x: x == 'C').astype(int)
    df['Qport'] = df['Embarked'].map(lambda x: x == 'Q').astype(int)
    df['Sport'] = df['Embarked'].map(lambda x: x == 'S').astype(int)

    # Fill in missing ages with medians for each gender-class combination
    median_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = df[(df['Gender'] == i) &
                                (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),
                'AgeFill'] = median_ages[i, j]

    # Fill in missing fares with the median
    df['FareFilled'] = df["Fare"]
    df.loc[df.Fare.isnull(), 'FareFilled'] = df.Fare.median()

    # Keep a record of which ages were originally null
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    # Interaction features
    # df['FamilySize'] = df['SibSp'] + df['Parch']
    # df['Age*Class'] = df.AgeFill * df.Pclass

    # Drop the useless features (strings, ticket number, incomplete, etc)
    dropped_features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare']
    df = df.drop(dropped_features, axis=1)
    df = df.dropna()

    return df

train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Remove passenger IDs and survival status from train data, because that's cheating
id_train = train_df.PassengerId.values
X = train_df.drop(['Survived', 'PassengerId'], axis=1).values
y = train_df['Survived'].values

# Create train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=.2)

# Format test data
id_test = test_df.PassengerId.values
X_test = test_df.drop('PassengerId', axis=1).values

clf = LogisticRegression(n_jobs=-1, max_iter=10**6)
clf = clf.fit(X_train, y_train)

print "Train accuracy: %f" % accuracy_score(y_train, clf.predict(X_train))
print "Valid accuracy: %f" % accuracy_score(y_valid, clf.predict(X_valid))

y_pred = clf.predict(X_test)
output_file = open("logreg_pred.csv", "wb")
writer = csv.writer(output_file)
writer.writerow(["PassengerId", "Survived"])
writer.writerows(zip(id_test, y_pred))
output_file.close()
print "Done."
