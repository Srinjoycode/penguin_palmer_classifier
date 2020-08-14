import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

penguins = pd.read_csv('penguins.csv')

df = penguins.copy()

# One Hot encoding of the  columns
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# value encoding of the target
target_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    return target_map[val]


df['species'] = df['species'].apply(target_encode)

# Seperating the labels labels and values
X = df.drop('species', axis=1)
Y = df['species']

# building a random forest Classifier

clf = RandomForestClassifier()

clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
