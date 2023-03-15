import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from MyNaiveBayes import NaiveBayes

data = np.load('spam.npy', allow_pickle=True)

y = data[:, 0]
X = data[:, 1]

# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def vectorize(X_train, X_test):
    # vectorize the training set
    count_vect = CountVectorizer(binary=True, max_features=1000)
    X_train = count_vect.fit_transform(X_train).toarray()
    X_test = count_vect.transform(X_test).toarray()
    return X_train, X_test

X_train, X_test = vectorize(X_train, X_test)

NB = NaiveBayes(X_train, y_train)
NB.fit(X_train, y_train)
y_pred = NB.predict(X_test)
conf_matrix = confusion_matrix(
    [int(i) for i in np.array(y_test)], 
    [int(i) for i in np.array(y_pred)]
)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(
            x=j,
            y=i,
            s=conf_matrix[i, j],
            va="center",
            ha="center",
            size="xx-large",
        )

fig.savefig("confusion_matrix_NB.png")
plt.show()
